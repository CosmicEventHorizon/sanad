package com.sanad.app.infrastructure.persistence.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "patients")
data class PatientEntity(
    @PrimaryKey val id: String,
    val fullName: String,
    val dateOfBirth: String,
    val phone: String,
    val village: String,
    val createdAt: Long,
)
