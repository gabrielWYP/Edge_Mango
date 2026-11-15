package com.edgemango.app.data

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.edgemango.app.core.ModelSidecar
import com.edgemango.app.core.ModelTask
import java.io.File

/** A model the app can load: a `.tflite` on disk plus its (possibly defaulted) sidecar. */
data class ModelRef(
    val id: String,
    val name: String,
    val file: File,
    val sidecar: ModelSidecar,
    val origin: String,
    /** Non-empty when the sidecar was missing or unreadable and defaults were used. */
    val warning: String = "",
) {
    val sizeBytes: Long get() = file.length()
    val sizeMb: Double get() = sizeBytes / (1024.0 * 1024.0)
}

/**
 * Where models come from, in the order they are listed:
 * 1. bundled in the APK (`assets/models/`), copied to internal storage so LiteRT can open them by path;
 * 2. `<app external files>/models/`  (`adb push` target, no storage permission needed);
 * 3. `<app files>/models/`           (where the "pick a file" import copies to).
 */
class ModelCatalog(private val context: Context) {
    private val bundledDir = File(context.filesDir, "bundled").apply { mkdirs() }
    private val importedDir = File(context.filesDir, "models").apply { mkdirs() }
    private val externalDir: File? = context.getExternalFilesDir("models")?.apply { mkdirs() }

    val externalDirPath: String get() = externalDir?.absolutePath ?: "(unavailable)"

    fun list(): List<ModelRef> {
        val out = mutableListOf<ModelRef>()
        val assetNames = context.assets.list("models")?.filter { it.endsWith(".tflite") }.orEmpty().sorted()
        for (name in assetNames) {
            val target = File(bundledDir, name)
            val assetSize = context.assets.openFd("models/$name").use { it.length }
            if (!target.exists() || target.length() != assetSize) {
                context.assets.open("models/$name").use { input -> target.outputStream().use { input.copyTo(it) } }
            }
            val sidecarText = runCatching { context.assets.open("models/${name.removeSuffix(".tflite")}.json").bufferedReader().use { it.readText() } }.getOrNull()
            out += build("asset:$name", name, target, sidecarText, "bundled")
        }
        for ((dir, origin) in listOf(externalDir to "adb / external files", importedDir to "imported")) {
            dir?.listFiles { f -> f.extension == "tflite" }?.sortedBy { it.name }?.forEach { f ->
                val sidecar = File(f.parentFile, f.nameWithoutExtension + ".json").takeIf { it.exists() }?.readText()
                out += build("file:${f.absolutePath}", f.name, f, sidecar, origin)
            }
        }
        return out
    }

    fun find(id: String): ModelRef? = list().firstOrNull { it.id == id }

    /** Copies the picked documents (a `.tflite` and, ideally, its `.json`) into app storage. Returns the copied names. */
    fun import(uris: List<Uri>): List<String> {
        val names = mutableListOf<String>()
        for (uri in uris) {
            val name = displayName(uri) ?: continue
            if (!(name.endsWith(".tflite") || name.endsWith(".json"))) continue
            context.contentResolver.openInputStream(uri)?.use { input ->
                File(importedDir, name).outputStream().use { input.copyTo(it) }
                names += name
            }
        }
        return names
    }

    fun delete(ref: ModelRef): Boolean = ref.origin == "imported" && ref.file.delete().also {
        File(ref.file.parentFile, ref.file.nameWithoutExtension + ".json").delete()
    }

    private fun build(id: String, name: String, file: File, sidecarText: String?, origin: String): ModelRef {
        val guessTask = if (name.contains("seg", ignoreCase = true)) ModelTask.SEGMENT else ModelTask.DETECT
        if (sidecarText == null) {
            return ModelRef(id, name, file, ModelSidecar.default(guessTask), origin, "no sidecar JSON: assumed ${guessTask.name.lowercase()}, 640 px, default decode settings")
        }
        return runCatching { ModelSidecar.parse(sidecarText) }.fold(
            onSuccess = { ModelRef(id, name, file, it, origin) },
            onFailure = { ModelRef(id, name, file, ModelSidecar.default(guessTask), origin, "unreadable sidecar (${it.message}): defaults used") },
        )
    }

    private fun displayName(uri: Uri): String? =
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        } ?: uri.lastPathSegment?.substringAfterLast('/')

    companion object {
        const val DEFAULT_ID = "asset:student_seg_nano_random_fp32.tflite"
    }
}
