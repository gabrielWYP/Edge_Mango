package com.edgemango.app.data

import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.util.zip.ZipFile

class SessionZipTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test
    fun zipsTheFolderUnderItsOwnNameKeepingSubfolders() {
        val dir = tmp.newFolder("session_20261121_101500")
        java.io.File(dir, "frames").mkdirs()
        java.io.File(dir, "frames/f_000000.jpg").writeBytes(byteArrayOf(1, 2, 3))
        java.io.File(dir, "predictions.jsonl").writeText("{}\n")
        java.io.File(dir, "session.json").writeText("{}")
        val out = SessionZip.zip(dir, java.io.File(tmp.root, "session_20261121_101500.zip"))
        ZipFile(out).use { z ->
            val names = z.entries().asSequence().map { it.name }.toList()
            assertEquals(
                listOf(
                    "session_20261121_101500/frames/f_000000.jpg",
                    "session_20261121_101500/predictions.jsonl",
                    "session_20261121_101500/session.json",
                ),
                names,
            )
            assertEquals(3, z.getInputStream(z.getEntry(names[0])).readBytes().size)
        }
    }
}
