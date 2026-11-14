package com.sanad.app.domain.form

interface IFormAssetCatalog {
    fun getAvailableForms(): List<FormAssetBundle>
    fun getForm(formId: String): FormAssetBundle?
}
