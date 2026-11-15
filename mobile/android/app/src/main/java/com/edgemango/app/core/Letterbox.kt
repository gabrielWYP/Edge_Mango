package com.edgemango.app.core

import kotlin.math.floor

/** Per-channel `(v / 255 - mean) / std`; the model does not normalize internally. */
data class Normalization(val mean: List<Double>, val std: List<Double>) {
    companion object {
        val IMAGENET = Normalization(listOf(0.485, 0.456, 0.406), listOf(0.229, 0.224, 0.225))
    }
}

object Letterbox {
    /**
     * Letterboxes `window` of [source] onto a `g.size` canvas filled with [padValue]: bilinear resize to
     * `newW x newH` (cv2.INTER_LINEAR, half-pixel centres, 8-bit rounding), centred.
     */
    fun toCanvas(
        source: RgbImage,
        window: IntBox,
        g: LetterboxGeometry,
        padValue: Int = 114,
        into: RgbImage = RgbImage(g.size, g.size, IntArray(g.size * g.size)),
    ): RgbImage {
        val size = g.size
        require(into.width == size && into.height == size) { "canvas must be ${size}x$size" }
        val pad = (0xFF shl 24) or (padValue shl 16) or (padValue shl 8) or padValue
        java.util.Arrays.fill(into.pixels, pad)

        val srcW = window.width
        val srcH = window.height
        val xScale = srcW.toDouble() / g.newW
        val yScale = srcH.toDouble() / g.newH
        val x0 = IntArray(g.newW)
        val x1 = IntArray(g.newW)
        val wx = FloatArray(g.newW)
        for (x in 0 until g.newW) {
            val fx = (x + 0.5) * xScale - 0.5
            val ix = floor(fx).toInt()
            val frac = (fx - ix).toFloat()
            x0[x] = ix.coerceIn(0, srcW - 1)
            x1[x] = (ix + 1).coerceIn(0, srcW - 1)
            wx[x] = frac
        }
        val rowA = FloatArray(g.newW * 3)
        val rowB = FloatArray(g.newW * 3)
        for (y in 0 until g.newH) {
            val fy = (y + 0.5) * yScale - 0.5
            val iy = floor(fy).toInt()
            val wy = (fy - iy).toFloat()
            val y0 = iy.coerceIn(0, srcH - 1) + window.y1
            val y1 = (iy + 1).coerceIn(0, srcH - 1) + window.y1
            horizontal(source, y0, window.x1, x0, x1, wx, rowA)
            horizontal(source, y1, window.x1, x0, x1, wx, rowB)
            val dst = (y + g.padY) * size + g.padX
            for (x in 0 until g.newW) {
                val o = x * 3
                val r = rowA[o] + (rowB[o] - rowA[o]) * wy
                val gr = rowA[o + 1] + (rowB[o + 1] - rowA[o + 1]) * wy
                val b = rowA[o + 2] + (rowB[o + 2] - rowA[o + 2]) * wy
                into.pixels[dst + x] = (0xFF shl 24) or (round8(r) shl 16) or (round8(gr) shl 8) or round8(b)
            }
        }
        return into
    }

    private fun round8(v: Float): Int = (v + 0.5f).toInt().coerceIn(0, 255)

    private fun horizontal(src: RgbImage, row: Int, xOffset: Int, x0: IntArray, x1: IntArray, wx: FloatArray, out: FloatArray) {
        val base = row * src.width + xOffset
        for (x in x0.indices) {
            val a = src.pixels[base + x0[x]]
            val b = src.pixels[base + x1[x]]
            val f = wx[x]
            val o = x * 3
            val ar = (a shr 16) and 0xFF
            val ag = (a shr 8) and 0xFF
            val ab = a and 0xFF
            out[o] = ar + (((b shr 16) and 0xFF) - ar) * f
            out[o + 1] = ag + (((b shr 8) and 0xFF) - ag) * f
            out[o + 2] = ab + ((b and 0xFF) - ab) * f
        }
    }

    /** Writes the canvas as an NHWC float32 RGB tensor, normalized. [out] must hold `width * height * 3` floats. */
    fun normalizeInto(canvas: RgbImage, out: FloatArray, norm: Normalization = Normalization.IMAGENET) {
        val lut = Array(3) { c ->
            FloatArray(256) { v ->
                val t = v / 255f
                ((t.toDouble() - norm.mean[c]) / norm.std[c]).toFloat()
            }
        }
        var o = 0
        for (i in 0 until canvas.width * canvas.height) {
            val p = canvas.pixels[i]
            out[o++] = lut[0][(p shr 16) and 0xFF]
            out[o++] = lut[1][(p shr 8) and 0xFF]
            out[o++] = lut[2][p and 0xFF]
        }
    }
}
