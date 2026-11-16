package com.sanad.app.infrastructure.ocr

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.channels.FileChannel

/** Loads large ONNX assets without first materializing the whole model on the Android heap. */
internal class AssetOnnxSessionLoader(
    private val context: Context,
    private val environment: OrtEnvironment,
) {
    fun load(assetPath: String): OrtSession = runCatching { loadMapped(assetPath) }
        .getOrElse { loadCachedFile(assetPath) }

    private fun loadMapped(assetPath: String): OrtSession {
        val descriptor = context.assets.openFd(assetPath)
        val input = FileInputStream(descriptor.fileDescriptor)
        return try {
            val model = input.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength,
            )
            environment.createSession(model, OrtSession.SessionOptions())
        } finally {
            input.close()
            runCatching { descriptor.close() }
        }
    }

    private fun loadCachedFile(assetPath: String): OrtSession {
        val modelFile = File(context.codeCacheDir, "onnx/${assetPath.replace('/', '_')}")
        if (!modelFile.isFile || modelFile.length() == 0L) {
            modelFile.parentFile?.mkdirs()
            val temporary = File(modelFile.parentFile, "${modelFile.name}.tmp")
            context.assets.open(assetPath).use { input ->
                FileOutputStream(temporary).use { output -> input.copyTo(output, DEFAULT_BUFFER_SIZE) }
            }
            check(temporary.renameTo(modelFile) || modelFile.isFile) {
                "Unable to prepare AI model $assetPath"
            }
        }
        return environment.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
    }
}
