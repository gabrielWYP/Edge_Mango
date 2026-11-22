package com.edgemango.app.data

import java.io.File
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** Zips a session folder (plain `java.util.zip`, JVM-testable). */
object SessionZip {
    /**
     * Writes [dir] into [out] with entries `<dir name>/<relative path>`, files sorted by path. JPEG frames barely
     * compress, so the fastest deflate level is used. Returns [out].
     */
    fun zip(dir: File, out: File): File {
        val root = dir.name
        ZipOutputStream(out.outputStream().buffered()).use { zip ->
            zip.setLevel(Deflater.BEST_SPEED)
            dir.walkTopDown().filter { it.isFile }.sortedBy { it.relativeTo(dir).path }.forEach { f ->
                zip.putNextEntry(ZipEntry("$root/${f.relativeTo(dir).path.replace(File.separatorChar, '/')}"))
                f.inputStream().use { it.copyTo(zip) }
                zip.closeEntry()
            }
        }
        return out
    }
}
