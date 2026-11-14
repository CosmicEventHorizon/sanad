package com.sanad.app.domain.model

data class Patient(
    val id: String,
    val fullName: String,
    val dateOfBirth: String,
    val phone: String,
    val village: String,
    val createdAt: Long = System.currentTimeMillis(),
)
