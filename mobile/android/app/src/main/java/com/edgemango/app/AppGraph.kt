package com.edgemango.app

import android.app.Application
import com.edgemango.app.core.RgbImage
import com.edgemango.app.data.ModelCatalog
import com.edgemango.app.data.ResultStore
import com.edgemango.app.data.SettingsRepository
import java.util.concurrent.atomic.AtomicReference

/** Hand-wired dependencies shared by the screens' view models (the app is small enough not to need a DI framework). */
class AppGraph(val app: Application) {
    val settings = SettingsRepository(app)
    val catalog = ModelCatalog(app)
    val results = ResultStore(app)

    /** The most recent camera frame (upright, full resolution), refreshed about once a second while the live screen runs. */
    val lastFrame = AtomicReference<RgbImage?>(null)
}

class EdgeMangoApplication : Application() {
    lateinit var graph: AppGraph
        private set

    override fun onCreate() {
        super.onCreate()
        graph = AppGraph(this)
    }
}
