package com.sanad.app.infrastructure.assets

import android.content.res.AssetManager
import com.sanad.app.domain.form.FormAssetBundle
import com.sanad.app.domain.form.IFormAssetCatalog
import org.json.JSONObject

/**
 * Discovers self-contained form bundles under assets/forms. New forms become
 * available by adding a directory containing spec.json and template.json.
 */
class AndroidFormAssetCatalog(
    private val assets: AssetManager,
) : IFormAssetCatalog {

    override fun getAvailableForms(): List<FormAssetBundle> =
        assets.list(FORMS_DIRECTORY).orEmpty()
            .mapNotNull(::loadBundle)
            .sortedBy(FormAssetBundle::formId)

    override fun getForm(formId: String): FormAssetBundle? {
        if (!SAFE_FORM_ID.matches(formId)) return null
        return loadBundle(formId)
    }

    private fun loadBundle(directoryName: String): FormAssetBundle? = runCatching {
        require(SAFE_FORM_ID.matches(directoryName))
        val specPath = "$FORMS_DIRECTORY/$directoryName/spec.json"
        val spec = assets.readJson(specPath)
        val formId = spec.getString("form_id")
        require(formId == directoryName) { "Form ID must match its asset directory" }

        val templatePath = spec.getString("template")
        val template = assets.readJson(templatePath)
        require(template.getString("form_id") == formId) { "Template and spec form IDs differ" }

        val pipeline = spec.getJSONObject("pipeline")
        FormAssetBundle(
            formId = formId,
            titleStringResource = template.getString("title_string_resource"),
            templatePath = templatePath,
            anchorsPath = spec.getString("anchors"),
            anchorModelPath = pipeline.getJSONObject("anchor_detection").getString("model"),
            handwritingRecognitionDirectory = pipeline.getJSONObject("handwriting_recognition").getString("model_directory"),
            numericRecognitionDirectory = pipeline.getJSONObject("numeric_recognition").getString("model_directory"),
        ).also(::validateAssets)
    }.getOrNull()

    private fun validateAssets(bundle: FormAssetBundle) {
        listOf(
            bundle.templatePath,
            bundle.anchorsPath,
            bundle.anchorModelPath,
        ).forEach { path -> assets.open(path).close() }
        require(!assets.list(bundle.handwritingRecognitionDirectory).isNullOrEmpty()) {
            "TrOCR asset directory is missing or empty"
        }
        require(!assets.list(bundle.numericRecognitionDirectory).isNullOrEmpty()) {
            "Numeric TrOCR asset directory is missing or empty"
        }
    }

    private fun AssetManager.readJson(path: String) =
        open(path).bufferedReader(Charsets.UTF_8).use { JSONObject(it.readText()) }

    private companion object {
        const val FORMS_DIRECTORY = "forms"
        val SAFE_FORM_ID = Regex("[a-z][a-z0-9_]*")
    }
}
