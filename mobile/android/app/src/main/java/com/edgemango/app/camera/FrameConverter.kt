package com.edgemango.app.camera

import androidx.camera.core.ImageProxy
import com.edgemango.app.core.RgbImage
import com.edgemango.app.core.Rotation
import java.nio.ByteOrder

/**
 * Turns an RGBA_8888 [ImageProxy] into an upright [RgbImage] (ARGB ints), applying `rotationDegrees` so that overlay
 * coordinates are in display orientation. The returned image wraps a reused buffer: it is only valid until the next call.
 */
class FrameConverter {
    private var raw = IntArray(0)
    private var upright = IntArray(0)

    fun toUpright(image: ImageProxy): RgbImage {
        val w = image.width
        val h = image.height
        val plane = image.planes[0]
        val buffer = plane.buffer.order(ByteOrder.LITTLE_ENDIAN)
        val rowInts = plane.rowStride / 4
        if (raw.size < w * h) raw = IntArray(w * h)
        if (upright.size < w * h) upright = IntArray(w * h)

        val ints = buffer.asIntBuffer()
        val row = IntArray(w)
        for (y in 0 until h) {
            ints.position(y * rowInts)
            ints.get(row, 0, w)
            val o = y * w
            for (x in 0 until w) {
                val p = row[x]
                // memory order R,G,B,A -> little-endian int 0xAABBGGRR -> ARGB
                raw[o + x] = (p and 0xFF00FF00.toInt()) or ((p and 0xFF) shl 16) or ((p shr 16) and 0xFF)
            }
        }
        return Rotation.clockwise(raw, w, h, image.imageInfo.rotationDegrees, upright)
    }
}

/** A copy of [image] that stays valid after the converter reuses its buffer. */
fun RgbImage.copyOf() = RgbImage(width, height, pixels.copyOf(width * height))
