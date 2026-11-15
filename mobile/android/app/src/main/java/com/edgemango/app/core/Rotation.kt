package com.edgemango.app.core

object Rotation {
    /**
     * Rotates a `w x h` pixel array clockwise by 0/90/180/270 degrees (what `ImageInfo.rotationDegrees` asks for to make
     * a camera frame upright). [dst] must hold `w * h` ints; for 0 degrees [src] is returned as it is.
     */
    fun clockwise(src: IntArray, w: Int, h: Int, degrees: Int, dst: IntArray): RgbImage {
        require(degrees % 90 == 0) { "rotation must be a multiple of 90, got $degrees" }
        val d = ((degrees % 360) + 360) % 360
        if (d == 0) return RgbImage(w, h, src)
        val dw = if (d == 90 || d == 270) h else w
        val dh = if (d == 90 || d == 270) w else h
        require(dst.size >= dw * dh) { "destination too small" }
        for (yd in 0 until dh) {
            val o = yd * dw
            for (xd in 0 until dw) {
                val s = when (d) {
                    90 -> (h - 1 - xd) * w + yd
                    180 -> (h - 1 - yd) * w + (w - 1 - xd)
                    else -> xd * w + (w - 1 - yd) // 270
                }
                dst[o + xd] = src[s]
            }
        }
        return RgbImage(dw, dh, dst)
    }
}
