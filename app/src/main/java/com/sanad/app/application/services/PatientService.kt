package com.sanad.app.application.services

import com.sanad.app.domain.model.Patient
import com.sanad.app.domain.repository.IPatientRepository

class PatientService(private val patientRepository: IPatientRepository) {
    suspend fun save(patient: Patient) {
        require(patient.id.isNotBlank()) { "Patient ID is required" }
        require(patient.fullName.isNotBlank()) { "Patient name is required" }
        patientRepository.save(patient)
    }

    suspend fun get(id: String) = patientRepository.findById(id)
}
