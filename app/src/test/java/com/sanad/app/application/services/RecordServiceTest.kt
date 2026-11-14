package com.sanad.app.application.services

import com.sanad.app.application.dto.RecordDraft
import com.sanad.app.domain.model.MedicalRecord
import com.sanad.app.domain.model.Patient
import com.sanad.app.domain.repository.IMedicalRecordRepository
import com.sanad.app.domain.repository.IPatientRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class RecordServiceTest {
    @Test
    fun `save verified persists patient and record`() = runTest {
        val patients = FakePatientRepository()
        val records = FakeRecordRepository()
        val service = RecordService(records, PatientService(patients))
        val id = service.saveVerified(RecordDraft(patientId = " P-7 ", patientName = " Mariam ", visitDate = "2026-11-14", bloodPressure = "118/76"))
        assertNotNull(id)
        assertEquals("P-7", patients.saved?.id)
        assertEquals("Mariam", records.saved?.patientName)
        assertEquals("118/76", records.saved?.bloodPressure)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `save verified requires patient id`() = runTest {
        RecordService(FakeRecordRepository(), PatientService(FakePatientRepository()))
            .saveVerified(RecordDraft(patientName = "Mariam", visitDate = "2026-11-14"))
    }
}

private class FakePatientRepository : IPatientRepository {
    var saved: Patient? = null
    override suspend fun save(patient: Patient) { saved = patient }
    override suspend fun findById(id: String) = saved?.takeIf { it.id == id }
    override fun observeAll(): Flow<List<Patient>> = flowOf(listOfNotNull(saved))
}

private class FakeRecordRepository : IMedicalRecordRepository {
    var saved: MedicalRecord? = null
    override suspend fun save(record: MedicalRecord) { saved = record }
    override suspend fun findById(id: String) = saved?.takeIf { it.id == id }
    override fun observeAll(): Flow<List<MedicalRecord>> = flowOf(listOfNotNull(saved))
    override fun search(query: String): Flow<List<MedicalRecord>> = observeAll()
}
