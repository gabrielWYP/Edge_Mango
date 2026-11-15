package com.edgemango.app.core

import java.io.InputStream
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.int
import kotlinx.serialization.json.double

/** Test helpers that read the fixtures produced by `tools/make_fixtures.py` from the classpath. */
object Fixtures {
    fun stream(path: String): InputStream =
        checkNotNull(Fixtures::class.java.classLoader.getResourceAsStream("fixtures/$path")) { "missing fixture $path" }

    fun bytes(path: String): ByteArray = stream(path).use { it.readBytes() }

    fun json(path: String): JsonObject =
        Json.parseToJsonElement(String(bytes(path), Charsets.UTF_8)).jsonObject

    /** Reads `*.rgb.gz`: int32-LE width, int32-LE height, RGB bytes (see `tools/make_fixtures.py`). */
    fun image(path: String): RgbImage {
        val raw = java.util.zip.GZIPInputStream(stream(path)).use { it.readBytes() }
        val bb = java.nio.ByteBuffer.wrap(raw).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val w = bb.getInt(0)
        val h = bb.getInt(4)
        val pixels = IntArray(w * h) { i ->
            val o = 8 + i * 3
            (0xFF shl 24) or ((raw[o].toInt() and 0xFF) shl 16) or ((raw[o + 1].toInt() and 0xFF) shl 8) or (raw[o + 2].toInt() and 0xFF)
        }
        return RgbImage(w, h, pixels)
    }

    fun floats(path: String): FloatArray {
        val buf = java.nio.ByteBuffer.wrap(bytes(path)).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buf.remaining()).also { buf.get(it) }
    }

    /** Dequantizes an int8 fixture: value = q / divisor. */
    fun quantized(path: String, divisor: Float): FloatArray = bytes(path).let { b -> FloatArray(b.size) { b[it] / divisor } }
}

fun JsonElement.ints(): List<Int> = jsonArray.map { it.jsonPrimitive.int }
fun JsonElement.doubles(): List<Double> = jsonArray.map { it.jsonPrimitive.double }
