package com.sanad.app.infrastructure.sync

import com.sanad.app.domain.model.MedicalRecord
import com.sanad.app.domain.service.IRecordSyncService

class PendingRecordSyncService : IRecordSyncService {
    override suspend fun sync(records: List<MedicalRecord>): Result<Int> =
        Result.failure(UnsupportedOperationException("Remote synchronization is not configured"))
}
