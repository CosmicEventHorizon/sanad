package com.sanad.app

import android.app.Application
import com.sanad.app.di.AppContainer

class SanadApplication : Application() {
    val container by lazy { AppContainer(this) }
}
