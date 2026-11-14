package com.sanad.app.infrastructure.ocr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import com.sanad.app.domain.model.RecognizedField
import com.sanad.app.domain.service.IFormProcessingService
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Offline first stage of the configured pipeline. YOLO OBB locates each
 * printed label, its local coordinate system is used to rectify the related
 * value area, and checkbox-like fields are classified by ink density. The
 * rectified handwriting crops are recognized by the configured TrOCR model;
 * failed inference remains empty and low-confidence for human review.
 */
class OnDeviceFormProcessingService(
    private val context: Context,
) : IFormProcessingService, AutoCloseable {
    private val environment by lazy { OrtEnvironment.getEnvironment() }
    private val sessions = mutableMapOf<String, OrtSession>()
    private val handwritingRecognizerDelegate = lazy { OnDeviceTrOcrRecognizer(context, environment) }
    private val handwritingRecognizer by handwritingRecognizerDelegate

    override suspend fun process(imageUri: String, formType: String): List<RecognizedField> =
        withContext(Dispatchers.Default) {
            val configuration = loadConfiguration(formType)
            val source = openBitmap(imageUri)
            val detected = runAnchorModel(source, configuration)
            val corrected = correctPerspective(source, detected, configuration.anchors)
            val bitmap = corrected.bitmap
            val detections = corrected.detections
            configuration.anchors.map { anchor ->
                val detection = detections[anchor.classId]
                when {
                    detection == null -> RecognizedField(anchor.fieldId, "", 0f)
                    anchor.type == "boolean" -> {
                        val crop = rectify(bitmap, detection, anchor.crop)
                        val ratio = inkRatio(crop, trimBorder = true)
                        val checked = ratio >= configuration.minimumInkRatio
                        RecognizedField(anchor.fieldId, checked.toString(), checkboxConfidence(ratio, configuration.minimumInkRatio))
                    }
                    anchor.type in setOf("single_choice", "multiple_choice") -> {
                        val scores = anchor.options.associateWith { option ->
                            inkRatio(rectify(bitmap, detection, option.crop), option.trimBorder)
                        }
                        val selected = if (anchor.type == "single_choice") {
                            scores.filter { (option, score) -> score >= (option.threshold ?: configuration.minimumInkRatio) }
                                .maxByOrNull { it.value }
                                ?.let { listOf(it.key) }
                                .orEmpty()
                        } else {
                            scores.filter { (option, score) -> score >= (option.threshold ?: configuration.minimumInkRatio) }.keys.toList()
                        }
                        val confidence = scores.minOfOrNull { (option, score) ->
                            checkboxConfidence(score, option.threshold ?: configuration.minimumInkRatio)
                        } ?: 0f
                        RecognizedField(anchor.fieldId, selected.joinToString(",") { it.id }, detection.confidence * confidence)
                    }
                    anchor.type in setOf("text", "number", "numeric") -> {
                        val crop = rectify(bitmap, detection, anchor.crop)
                        val directory = if (anchor.type in setOf("number", "numeric")) {
                            configuration.numericModelDirectory
                        } else configuration.textModelDirectory
                        runCatching { handwritingRecognizer.recognize(crop, directory) }
                            .fold(
                                onSuccess = {
                                    val text = if (anchor.type in setOf("number", "numeric")) {
                                        NUMERIC_PATTERN.find(it.text.replace(" ", ""))?.value.orEmpty()
                                    } else it.text
                                    RecognizedField(anchor.fieldId, text, detection.confidence * it.confidence)
                                },
                                onFailure = { RecognizedField(anchor.fieldId, "", detection.confidence * 0.25f) },
                            )
                    }
                    else -> RecognizedField(anchor.fieldId, "", detection.confidence * 0.25f)
                }
            }
        }

    private fun loadConfiguration(formType: String): Configuration {
        require(SAFE_ID.matches(formType)) { "Invalid form type" }
        val spec = context.assets.readJson("forms/$formType/spec.json")
        val pipeline = spec.getJSONObject("pipeline")
        val detector = pipeline.getJSONObject("anchor_detection")
        val checkbox = pipeline.getJSONObject("checkbox_recognition")
        val anchors = context.assets.readJson(spec.getString("anchors"))
            .getJSONArray("classes")
        return Configuration(
            modelPath = detector.getString("model"),
            inputSize = detector.getJSONObject("input").getInt("width"),
            confidenceThreshold = detector.getDouble("confidence_threshold").toFloat(),
            minimumInkRatio = checkbox.getDouble("minimum_ink_ratio").toFloat(),
            textModelDirectory = pipeline.getJSONObject("handwriting_recognition").getString("model_directory"),
            numericModelDirectory = pipeline.getJSONObject("numeric_recognition").getString("model_directory"),
            anchors = (0 until anchors.length()).map { index ->
                val item = anchors.getJSONObject(index)
                val crop = item.getJSONArray("relative_crop")
                val canonical = item.getJSONArray("canonical_anchor")
                Anchor(
                    classId = item.getInt("class_id"),
                    fieldId = item.getString("field_id"),
                    type = item.getString("field_type"),
                    crop = FloatArray(4) { crop.getDouble(it).toFloat() },
                    canonical = FloatArray(4) { canonical.getDouble(it).toFloat() },
                    options = item.optJSONArray("options")?.let { options ->
                        (0 until options.length()).map { optionIndex ->
                            val option = options.getJSONObject(optionIndex)
                            val optionCrop = option.getJSONArray("relative_crop")
                            ChoiceOption(
                                id = option.getString("id"),
                                crop = FloatArray(4) { optionCrop.getDouble(it).toFloat() },
                                threshold = if (option.has("ink_threshold")) option.getDouble("ink_threshold").toFloat() else null,
                                trimBorder = option.optBoolean("trim_border", true),
                            )
                        }
                    }.orEmpty(),
                )
            },
        )
    }

    private fun runAnchorModel(bitmap: Bitmap, config: Configuration): Map<Int, Detection> {
        val prepared = letterbox(bitmap, config.inputSize)
        val data = FloatArray(3 * config.inputSize * config.inputSize)
        var redOffset = 0
        var greenOffset = config.inputSize * config.inputSize
        var blueOffset = greenOffset * 2
        val pixels = IntArray(config.inputSize * config.inputSize)
        prepared.bitmap.getPixels(pixels, 0, config.inputSize, 0, 0, config.inputSize, config.inputSize)
        pixels.forEach { pixel ->
            data[redOffset++] = Color.red(pixel) / 255f
            data[greenOffset++] = Color.green(pixel) / 255f
            data[blueOffset++] = Color.blue(pixel) / 255f
        }

        val session = sessions.getOrPut(config.modelPath) {
            val bytes = context.assets.open(config.modelPath).use { it.readBytes() }
            environment.createSession(bytes, OrtSession.SessionOptions())
        }
        val inputName = session.inputNames.first()
        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(data),
            longArrayOf(1, 3, config.inputSize.toLong(), config.inputSize.toLong()),
        ).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val tensor = result[0] as OnnxTensor
                val shape = tensor.info.shape
                require(shape.size == 3) { "Unexpected YOLO output shape" }
                val channels = shape[1].toInt()
                val candidates = shape[2].toInt()
                val classCount = channels - 5
                require(classCount == config.anchors.size) { "YOLO class count does not match anchors.json" }
                val values = tensor.floatBuffer.let { buffer: FloatBuffer ->
                    FloatArray(buffer.remaining()).also(buffer::get)
                }
                val best = mutableMapOf<Int, Detection>()
                for (candidate in 0 until candidates) {
                    var classId = 0
                    var confidence = values[4 * candidates + candidate]
                    for (index in 1 until classCount) {
                        val score = values[(4 + index) * candidates + candidate]
                        if (score > confidence) {
                            confidence = score
                            classId = index
                        }
                    }
                    if (confidence < config.confidenceThreshold) continue
                    val angle = values[(channels - 1) * candidates + candidate]
                    val detection = Detection(
                        centerX = (values[candidate] - prepared.left) / prepared.scale,
                        centerY = (values[candidates + candidate] - prepared.top) / prepared.scale,
                        width = values[2 * candidates + candidate] / prepared.scale,
                        height = values[3 * candidates + candidate] / prepared.scale,
                        angle = angle,
                        confidence = confidence,
                    )
                    if (confidence > (best[classId]?.confidence ?: -1f)) best[classId] = detection
                }
                return best
            }
        }
    }

    private fun rectify(source: Bitmap, anchor: Detection, relative: FloatArray): Bitmap {
        val corners = localCorners(anchor, relative)
        val width = max(32, distance(corners, 0, 1).toInt())
        val height = max(24, distance(corners, 1, 2).toInt())
        val destination = floatArrayOf(0f, 0f, width.toFloat(), 0f, width.toFloat(), height.toFloat(), 0f, height.toFloat())
        val matrix = Matrix().apply { setPolyToPoly(corners, 0, destination, 0, 4) }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { output ->
            Canvas(output).drawBitmap(source, matrix, Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG))
        }
    }

    private fun correctPerspective(source: Bitmap, detections: Map<Int, Detection>, anchors: List<Anchor>): CorrectedForm {
        val referenceIds = listOf(0, 2, 7, 9)
        if (referenceIds.any { it !in detections }) return CorrectedForm(source, detections)
        val sourcePoints = referenceIds.flatMap { id ->
            detections.getValue(id).let { listOf(it.centerX, it.centerY) }
        }.toFloatArray()
        val byId = anchors.associateBy(Anchor::classId)
        val destinationPoints = referenceIds.flatMap { id ->
            val box = byId.getValue(id).canonical
            listOf((box[0] + box[2]) * 0.5f * source.width, (box[1] + box[3]) * 0.5f * source.height)
        }.toFloatArray()
        val matrix = Matrix()
        if (!matrix.setPolyToPoly(sourcePoints, 0, destinationPoints, 0, 4)) return CorrectedForm(source, detections)
        val bitmap = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        Canvas(bitmap).apply {
            drawColor(Color.WHITE)
            drawBitmap(source, matrix, Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG))
        }
        val canonicalDetections = anchors.associate { anchor ->
            val box = anchor.canonical
            anchor.classId to Detection(
                centerX = (box[0] + box[2]) * 0.5f * source.width,
                centerY = (box[1] + box[3]) * 0.5f * source.height,
                width = (box[2] - box[0]) * source.width,
                height = (box[3] - box[1]) * source.height,
                angle = 0f,
                confidence = detections[anchor.classId]?.confidence ?: 0f,
            )
        }.filterValues { it.confidence > 0f }
        return CorrectedForm(bitmap, canonicalDetections)
    }

    private fun localCorners(anchor: Detection, crop: FloatArray): FloatArray {
        val ux = cos(anchor.angle)
        val uy = sin(anchor.angle)
        val vx = -uy
        val vy = ux
        fun point(x: Float, y: Float): Pair<Float, Float> =
            (anchor.centerX + (x - 0.5f) * anchor.width * ux + (y - 0.5f) * anchor.height * vx) to
                (anchor.centerY + (x - 0.5f) * anchor.width * uy + (y - 0.5f) * anchor.height * vy)
        val points = listOf(point(crop[0], crop[1]), point(crop[2], crop[1]), point(crop[2], crop[3]), point(crop[0], crop[3]))
        return points.flatMap { listOf(it.first, it.second) }.toFloatArray()
    }

    private fun inkRatio(bitmap: Bitmap, trimBorder: Boolean): Float {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val marginX = if (trimBorder) max(1, bitmap.width / 5) else 0
        val marginY = if (trimBorder) max(1, bitmap.height / 5) else 0
        val gray = buildList {
            for (y in marginY until bitmap.height - marginY) for (x in marginX until bitmap.width - marginX) {
                val pixel = pixels[y * bitmap.width + x]
                add(Color.red(pixel) * 0.299f + Color.green(pixel) * 0.587f + Color.blue(pixel) * 0.114f)
            }
        }
        val threshold = min(190f, gray.average().toFloat() * 0.78f)
        return gray.count { it < threshold }.toFloat() / max(1, gray.size)
    }

    private fun checkboxConfidence(ratio: Float, threshold: Float): Float =
        (0.5f + min(0.49f, kotlin.math.abs(ratio - threshold) / max(threshold, 0.01f))).coerceIn(0f, 0.99f)

    private fun letterbox(source: Bitmap, size: Int): PreparedBitmap {
        val scale = min(size.toFloat() / source.width, size.toFloat() / source.height)
        val width = source.width * scale
        val height = source.height * scale
        val left = (size - width) / 2f
        val top = (size - height) / 2f
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(bitmap).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(source, null, android.graphics.RectF(left, top, left + width, top + height), Paint(Paint.FILTER_BITMAP_FLAG))
        }
        return PreparedBitmap(bitmap, scale, left, top)
    }

    private fun openBitmap(value: String): Bitmap {
        val uri = Uri.parse(value)
        return context.contentResolver.openInputStream(uri)?.use(BitmapFactory::decodeStream)
            ?: error("Unable to open captured form image")
    }

    override fun close() {
        sessions.values.forEach(OrtSession::close)
        sessions.clear()
        if (handwritingRecognizerDelegate.isInitialized()) handwritingRecognizer.close()
    }

    private fun android.content.res.AssetManager.readJson(path: String) =
        open(path).bufferedReader().use { JSONObject(it.readText()) }

    private fun distance(points: FloatArray, firstPoint: Int, secondPoint: Int): Float {
        val dx = points[firstPoint * 2] - points[secondPoint * 2]
        val dy = points[firstPoint * 2 + 1] - points[secondPoint * 2 + 1]
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    private data class Configuration(
        val modelPath: String,
        val inputSize: Int,
        val confidenceThreshold: Float,
        val minimumInkRatio: Float,
        val textModelDirectory: String,
        val numericModelDirectory: String,
        val anchors: List<Anchor>,
    )
    private data class Anchor(
        val classId: Int,
        val fieldId: String,
        val type: String,
        val crop: FloatArray,
        val canonical: FloatArray,
        val options: List<ChoiceOption>,
    )
    private data class ChoiceOption(val id: String, val crop: FloatArray, val threshold: Float?, val trimBorder: Boolean)
    private data class Detection(val centerX: Float, val centerY: Float, val width: Float, val height: Float, val angle: Float, val confidence: Float)
    private data class PreparedBitmap(val bitmap: Bitmap, val scale: Float, val left: Float, val top: Float)
    private data class CorrectedForm(val bitmap: Bitmap, val detections: Map<Int, Detection>)

    private companion object {
        val SAFE_ID = Regex("[a-z][a-z0-9_]*")
        val NUMERIC_PATTERN = Regex("[+-]?\\d+(?:[.,/:-]\\d+)*(?:%)?")
    }
}
