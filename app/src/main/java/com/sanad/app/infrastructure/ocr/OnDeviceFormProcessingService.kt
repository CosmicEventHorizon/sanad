package com.sanad.app.infrastructure.ocr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
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
 * Offline form pipeline. A form-specific YOLO OBB locates printed labels and a shared YOLO OBB
 * independently locates handwriting. Label-relative regions are used only to assign handwriting
 * detections to fields; TrOCR receives a padded crop of the detected handwriting polygon itself.
 * Checkbox-like fields remain label-relative and are classified by ink density.
 */
class OnDeviceFormProcessingService(
    private val context: Context,
) : IFormProcessingService, AutoCloseable {
    private val environment by lazy { OrtEnvironment.getEnvironment() }
    private val sessions = mutableMapOf<String, OrtSession>()
    private val sessionLoader = AssetOnnxSessionLoader(context, environment)
    private val handwritingRecognizerDelegate = lazy { OnDeviceTrOcrRecognizer(context, environment) }
    private val handwritingRecognizer by handwritingRecognizerDelegate

    override suspend fun process(imageUri: String, formType: String): List<RecognizedField> =
        withContext(Dispatchers.Default) {
            val configuration = loadConfiguration(formType)
            val source = openBitmap(imageUri)
            val anchors = runObbModel(source, configuration.anchorDetector, configuration.anchors.size)
                .groupBy(OrientedBox::classId)
                .mapValues { (_, detections) -> detections.maxBy(OrientedBox::confidence) }
            require(anchors.isNotEmpty()) {
                "No pregnancy-monitoring form labels were detected. Fill the frame with the complete form and try again."
            }
            val handwriting = runObbModel(source, configuration.handwritingDetector, expectedClassCount = 1)
            val textFields = configuration.anchors.mapNotNull { anchor ->
                val detectedAnchor = anchors[anchor.classId]
                if (detectedAnchor != null && anchor.type in TEXT_FIELD_TYPES) {
                    AnchoredFieldRegion(anchor.fieldId, detectedAnchor, anchor.crop)
                } else null
            }
            val handwritingMatches = HandwritingFieldMatcher.match(textFields, handwriting)
            configuration.anchors.map { anchor ->
                val detection = anchors[anchor.classId]
                when {
                    detection == null -> RecognizedField(anchor.fieldId, "", 0f)
                    anchor.type == "boolean" -> {
                        val crop = rectify(source, detection, anchor.crop)
                        val ratio = inkRatio(crop, trimBorder = true)
                        val checked = ratio >= configuration.minimumInkRatio
                        RecognizedField(anchor.fieldId, checked.toString(), checkboxConfidence(ratio, configuration.minimumInkRatio))
                    }
                    anchor.type in setOf("single_choice", "multiple_choice") -> {
                        val scores = anchor.options.associateWith { option ->
                            inkRatio(rectify(source, detection, option.crop), option.trimBorder)
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
                    anchor.type in TEXT_FIELD_TYPES -> {
                        val handwritingDetection = handwritingMatches[anchor.fieldId]
                            ?: return@map RecognizedField(anchor.fieldId, "", 0f)
                        val crop = rectifyDetection(source, handwritingDetection, configuration.handwritingCropPadding)
                        val directory = if (anchor.type in setOf("number", "numeric")) {
                            configuration.numericModelDirectory
                        } else configuration.textModelDirectory
                        runCatching { handwritingRecognizer.recognize(crop, directory) }
                            .fold(
                                onSuccess = {
                                    val text = if (anchor.type in setOf("number", "numeric")) {
                                        NUMERIC_PATTERN.find(it.text.replace(" ", ""))?.value.orEmpty()
                                    } else it.text
                                    val localizationConfidence = min(detection.confidence, handwritingDetection.confidence)
                                    RecognizedField(anchor.fieldId, text, localizationConfidence * it.confidence)
                                },
                                onFailure = { RecognizedField(anchor.fieldId, "", 0f) },
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
        val handwritingDetector = pipeline.getJSONObject("handwriting_detection")
        val checkbox = pipeline.getJSONObject("checkbox_recognition")
        val anchors = context.assets.readJson(spec.getString("anchors"))
            .getJSONArray("classes")
        return Configuration(
            anchorDetector = detector.toDetectorConfiguration(),
            handwritingDetector = handwritingDetector.toDetectorConfiguration(),
            handwritingCropPadding = handwritingDetector.optDouble("crop_padding", 0.15).toFloat(),
            minimumInkRatio = checkbox.getDouble("minimum_ink_ratio").toFloat(),
            textModelDirectory = pipeline.getJSONObject("handwriting_recognition").getString("model_directory"),
            numericModelDirectory = pipeline.getJSONObject("numeric_recognition").getString("model_directory"),
            anchors = (0 until anchors.length()).map { index ->
                val item = anchors.getJSONObject(index)
                val crop = item.getJSONArray("relative_crop")
                Anchor(
                    classId = item.getInt("class_id"),
                    fieldId = item.getString("field_id"),
                    type = item.getString("field_type"),
                    crop = FloatArray(4) { crop.getDouble(it).toFloat() },
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

    private fun JSONObject.toDetectorConfiguration() = DetectorConfiguration(
        modelPath = getString("model"),
        inputSize = getJSONObject("input").getInt("width"),
        confidenceThreshold = getDouble("confidence_threshold").toFloat(),
        iouThreshold = optDouble("iou_threshold", 0.45).toFloat(),
        maximumDetections = optInt("maximum_detections", 150),
    )

    private fun runObbModel(bitmap: Bitmap, detector: DetectorConfiguration, expectedClassCount: Int): List<OrientedBox> {
        val prepared = letterbox(bitmap, detector.inputSize)
        val data = FloatArray(3 * detector.inputSize * detector.inputSize)
        var redOffset = 0
        var greenOffset = detector.inputSize * detector.inputSize
        var blueOffset = greenOffset * 2
        val pixels = IntArray(detector.inputSize * detector.inputSize)
        prepared.bitmap.getPixels(pixels, 0, detector.inputSize, 0, 0, detector.inputSize, detector.inputSize)
        pixels.forEach { pixel ->
            data[redOffset++] = Color.red(pixel) / 255f
            data[greenOffset++] = Color.green(pixel) / 255f
            data[blueOffset++] = Color.blue(pixel) / 255f
        }

        val session = sessions.getOrPut(detector.modelPath) {
            sessionLoader.load(detector.modelPath)
        }
        val inputName = session.inputNames.first()
        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(data),
            longArrayOf(1, 3, detector.inputSize.toLong(), detector.inputSize.toLong()),
        ).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val tensor = result[0] as OnnxTensor
                val shape = tensor.info.shape
                require(shape.size == 3) { "Unexpected YOLO output shape" }
                val channels = shape[1].toInt()
                val candidates = shape[2].toInt()
                val classCount = channels - 5
                require(classCount == expectedClassCount) { "YOLO class count does not match its configuration" }
                val values = tensor.floatBuffer.let { buffer: FloatBuffer ->
                    FloatArray(buffer.remaining()).also(buffer::get)
                }
                val detections = mutableListOf<OrientedBox>()
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
                    if (confidence < detector.confidenceThreshold) continue
                    val angle = values[(channels - 1) * candidates + candidate]
                    val detection = OrientedBox(
                        centerX = (values[candidate] - prepared.left) / prepared.scale,
                        centerY = (values[candidates + candidate] - prepared.top) / prepared.scale,
                        width = values[2 * candidates + candidate] / prepared.scale,
                        height = values[3 * candidates + candidate] / prepared.scale,
                        angle = angle,
                        confidence = confidence,
                        classId = classId,
                    )
                    if (detection.width >= 2f && detection.height >= 2f) detections += detection
                }
                return nonMaximumSuppression(detections, detector.iouThreshold, detector.maximumDetections)
            }
        }
    }

    private fun rectify(source: Bitmap, anchor: OrientedBox, relative: FloatArray): Bitmap {
        val corners = localCorners(anchor, relative)
        val width = max(32, distance(corners, 0, 1).toInt())
        val height = max(24, distance(corners, 1, 2).toInt())
        val destination = floatArrayOf(0f, 0f, width.toFloat(), 0f, width.toFloat(), height.toFloat(), 0f, height.toFloat())
        val matrix = Matrix().apply { setPolyToPoly(corners, 0, destination, 0, 4) }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { output ->
            Canvas(output).drawBitmap(source, matrix, Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG))
        }
    }

    private fun rectifyDetection(source: Bitmap, detection: OrientedBox, padding: Float): Bitmap {
        val horizontalPadding = max(4f, detection.height * padding)
        val verticalPadding = max(3f, detection.height * padding)
        val padded = detection.copy(
            width = detection.width + horizontalPadding * 2f,
            height = detection.height + verticalPadding * 2f,
        )
        return rectify(source, padded, floatArrayOf(0f, 0f, 1f, 1f))
    }

    private fun localCorners(anchor: OrientedBox, crop: FloatArray): FloatArray {
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

    private fun nonMaximumSuppression(
        candidates: List<OrientedBox>,
        threshold: Float,
        maximumDetections: Int,
    ): List<OrientedBox> {
        val sorted = candidates.sortedByDescending(OrientedBox::confidence)
        val kept = mutableListOf<OrientedBox>()
        for (candidate in sorted) {
            if (kept.size >= maximumDetections) break
            if (kept.none { it.classId == candidate.classId && axisAlignedIou(it, candidate) > threshold }) kept += candidate
        }
        return kept
    }

    /** Conservative NMS approximation; final crops still preserve the OBB angle. */
    private fun axisAlignedIou(first: OrientedBox, second: OrientedBox): Float {
        val firstLeft = first.centerX - first.width * 0.5f
        val firstTop = first.centerY - first.height * 0.5f
        val firstRight = first.centerX + first.width * 0.5f
        val firstBottom = first.centerY + first.height * 0.5f
        val secondLeft = second.centerX - second.width * 0.5f
        val secondTop = second.centerY - second.height * 0.5f
        val secondRight = second.centerX + second.width * 0.5f
        val secondBottom = second.centerY + second.height * 0.5f
        val intersection = max(0f, min(firstRight, secondRight) - max(firstLeft, secondLeft)) *
            max(0f, min(firstBottom, secondBottom) - max(firstTop, secondTop))
        val union = first.width * first.height + second.width * second.height - intersection
        return if (union <= 0f) 0f else intersection / union
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
        val bitmap = context.contentResolver.openInputStream(uri)?.use(BitmapFactory::decodeStream)
            ?: error("Unable to open captured form image")
        val orientation = context.contentResolver.openInputStream(uri)?.use { input ->
            ExifInterface(input).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } ?: ExifInterface.ORIENTATION_NORMAL
        val rotation = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90, ExifInterface.ORIENTATION_TRANSPOSE -> 90f
            ExifInterface.ORIENTATION_ROTATE_180, ExifInterface.ORIENTATION_FLIP_VERTICAL -> 180f
            ExifInterface.ORIENTATION_ROTATE_270, ExifInterface.ORIENTATION_TRANSVERSE -> 270f
            else -> 0f
        }
        val flipHorizontal = orientation in setOf(
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL,
            ExifInterface.ORIENTATION_TRANSPOSE,
            ExifInterface.ORIENTATION_TRANSVERSE,
        )
        if (rotation == 0f && !flipHorizontal) return bitmap
        val matrix = Matrix().apply {
            if (flipHorizontal) postScale(-1f, 1f)
            if (rotation != 0f) postRotate(rotation)
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
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
        val anchorDetector: DetectorConfiguration,
        val handwritingDetector: DetectorConfiguration,
        val handwritingCropPadding: Float,
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
        val options: List<ChoiceOption>,
    )
    private data class DetectorConfiguration(
        val modelPath: String,
        val inputSize: Int,
        val confidenceThreshold: Float,
        val iouThreshold: Float,
        val maximumDetections: Int,
    )
    private data class ChoiceOption(val id: String, val crop: FloatArray, val threshold: Float?, val trimBorder: Boolean)
    private data class PreparedBitmap(val bitmap: Bitmap, val scale: Float, val left: Float, val top: Float)

    private companion object {
        val SAFE_ID = Regex("[a-z][a-z0-9_]*")
        val NUMERIC_PATTERN = Regex("[+-]?\\d+(?:[.,/:-]\\d+)*(?:%)?")
        val TEXT_FIELD_TYPES = setOf("text", "number", "numeric")
    }
}
