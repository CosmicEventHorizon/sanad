package com.sanad.app.infrastructure.ocr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.exp
import kotlin.math.max

data class TrOcrResult(val text: String, val confidence: Float)

/** Greedy, fully offline TrOCR encoder/decoder inference through ONNX Runtime. */
class OnDeviceTrOcrRecognizer(
    private val context: Context,
    private val environment: OrtEnvironment,
) : AutoCloseable {
    private val sessions = mutableMapOf<String, OrtSession>()
    private val vocabularies = mutableMapOf<String, Map<Int, String>>()

    fun recognize(bitmap: Bitmap, modelDirectory: String, maxLength: Int = 32): TrOcrResult {
        val encoderDirectory = if (assetExists("$modelDirectory/encoder_model.onnx")) modelDirectory else TEXT_MODEL_DIRECTORY
        val encoder = session("$encoderDirectory/encoder_model.onnx")
        val decoder = session("$modelDirectory/decoder_model.onnx")
        val pixels = preprocess(bitmap)
        val hidden = OnnxTensor.createTensor(environment, FloatBuffer.wrap(pixels), longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())).use { input ->
            encoder.run(mapOf(encoder.inputNames.first() to input)).use { result ->
                val output = result[0] as OnnxTensor
                val shape = output.info.shape.clone()
                val values = output.floatBuffer.let { buffer: FloatBuffer -> FloatArray(buffer.remaining()).also(buffer::get) }
                HiddenState(values, shape)
            }
        }

        val generated = mutableListOf(START_TOKEN_ID)
        val probabilities = mutableListOf<Float>()
        for (step in 1 until maxLength) {
            val ids = generated.map(Int::toLong).toLongArray()
            var reachedEnd = false
            OnnxTensor.createTensor(environment, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong())).use { idTensor ->
                OnnxTensor.createTensor(environment, FloatBuffer.wrap(hidden.values), hidden.shape).use { encoderTensor ->
                    val inputs = decoder.inputNames.associateWith { name ->
                        when (name) {
                            "input_ids", "decoder_input_ids" -> idTensor
                            "encoder_hidden_states", "encoder_outputs" -> encoderTensor
                            else -> error("Unsupported TrOCR decoder input: $name")
                        }
                    }
                    decoder.run(inputs).use { result ->
                        val logits = result[0] as OnnxTensor
                        val shape = logits.info.shape
                        val vocabularySize = shape.last().toInt()
                        val values = logits.floatBuffer.let { buffer: FloatBuffer -> FloatArray(buffer.remaining()).also(buffer::get) }
                        val offset = values.size - vocabularySize
                        var token = 0
                        var best = Float.NEGATIVE_INFINITY
                        for (index in 0 until vocabularySize) {
                            if (values[offset + index] > best) {
                                best = values[offset + index]
                                token = index
                            }
                        }
                        var denominator = 0.0
                        for (index in 0 until vocabularySize) denominator += exp((values[offset + index] - best).toDouble())
                        probabilities += (1.0 / max(denominator, 1e-12)).toFloat()
                        if (token == END_TOKEN_ID) reachedEnd = true else generated += token
                    }
                }
            }
            if (reachedEnd) break
        }
        val vocabulary = vocabularies.getOrPut(modelDirectory) { loadVocabulary(modelDirectory) }
        val text = decodeByteBpe(generated.drop(1).mapNotNull(vocabulary::get))
        val confidence = if (probabilities.isEmpty()) 0f else probabilities.average().toFloat().coerceIn(0f, 1f)
        return TrOcrResult(text.trim(), confidence)
    }

    private fun preprocess(source: Bitmap): FloatArray {
        val bitmap = Bitmap.createScaledBitmap(source, IMAGE_SIZE, IMAGE_SIZE, true)
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        bitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        val result = FloatArray(3 * pixels.size)
        pixels.forEachIndexed { index, pixel ->
            result[index] = Color.red(pixel) / 127.5f - 1f
            result[pixels.size + index] = Color.green(pixel) / 127.5f - 1f
            result[pixels.size * 2 + index] = Color.blue(pixel) / 127.5f - 1f
        }
        return result
    }

    private fun loadVocabulary(directory: String): Map<Int, String> {
        val json = context.assets.open("$directory/vocab.json").bufferedReader().use { JSONObject(it.readText()) }
        return json.keys().asSequence().associate { token -> json.getInt(token) to token }
    }

    private fun decodeByteBpe(tokens: List<String>): String {
        val pieces = tokens.filterNot { it.startsWith("<") && it.endsWith(">") }
        if (pieces.any { '▁' in it }) return pieces.joinToString("").replace('▁', ' ')
        val encoded = pieces.joinToString("")
        val reverse = byteUnicode().entries.associate { it.value to it.key }
        val bytes = encoded.mapNotNull(reverse::get).map(Int::toByte).toByteArray()
        return bytes.toString(Charsets.UTF_8)
    }

    private fun byteUnicode(): Map<Int, Char> {
        val bytes = ((33..126) + (161..172) + (174..255)).toMutableList()
        val chars = bytes.toMutableList()
        var extra = 0
        for (value in 0..255) if (value !in bytes) {
            bytes += value
            chars += 256 + extra++
        }
        return bytes.indices.associate { bytes[it] to chars[it].toChar() }
    }

    private fun session(path: String): OrtSession = sessions.getOrPut(path) {
        context.assets.open(path).use { environment.createSession(it.readBytes(), OrtSession.SessionOptions()) }
    }

    private fun assetExists(path: String): Boolean = runCatching { context.assets.open(path).close() }.isSuccess

    override fun close() {
        sessions.values.forEach(OrtSession::close)
        sessions.clear()
    }

    private data class HiddenState(val values: FloatArray, val shape: LongArray)

    private companion object {
        const val IMAGE_SIZE = 384
        const val START_TOKEN_ID = 0
        const val END_TOKEN_ID = 2
        const val TEXT_MODEL_DIRECTORY = "models/trocr"
    }
}
