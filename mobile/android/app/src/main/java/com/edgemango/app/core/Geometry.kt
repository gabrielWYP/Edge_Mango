package com.edgemango.app.core

/** An 8-bit RGB image packed as 0xAARRGGBB ints (the layout of `Bitmap.getPixels`). Alpha is ignored. */
class RgbImage(val width: Int, val height: Int, val pixels: IntArray) {
    init {
        require(pixels.size >= width * height) { "pixel buffer too small for ${width}x$height" }
    }
}

/** Integer pixel box, `x2`/`y2` exclusive. */
data class IntBox(val x1: Int, val y1: Int, val x2: Int, val y2: Int) {
    val width: Int get() = x2 - x1
    val height: Int get() = y2 - y1
    val area: Long get() = maxOf(0, width).toLong() * maxOf(0, height)

    fun iou(other: IntBox): Double {
        val iw = minOf(x2, other.x2) - maxOf(x1, other.x1)
        val ih = minOf(y2, other.y2) - maxOf(y1, other.y1)
        if (iw <= 0 || ih <= 0) return 0.0
        val inter = iw.toLong() * ih
        return inter.toDouble() / (area + other.area - inter)
    }
}

/** A detection box in source-frame pixels (float, clipped to the frame). */
data class SourceBox(val x1: Double, val y1: Double, val x2: Double, val y2: Double)

/**
 * The letterbox of a crop (or a whole frame) onto a square `size` canvas, exactly like
 * `src.training.dataset.letterbox`: `scale = size / max(w, h)`, `new = floor(dim * scale)`, centred.
 */
data class LetterboxGeometry(
    val srcW: Int,
    val srcH: Int,
    val size: Int,
    val scale: Double,
    val newW: Int,
    val newH: Int,
    val padX: Int,
    val padY: Int,
) {
    /** Canvas pixel -> pixel of the source crop (not clipped). */
    fun toSource(x: Double, y: Double): Pair<Double, Double> = Pair((x - padX) / scale, (y - padY) / scale)

    companion object {
        fun of(srcW: Int, srcH: Int, size: Int): LetterboxGeometry {
            require(srcW > 0 && srcH > 0) { "empty source ${srcW}x$srcH" }
            val scale = size.toDouble() / maxOf(srcW, srcH)
            val newW = (srcW * scale).toInt()
            val newH = (srcH * scale).toInt()
            return LetterboxGeometry(srcW, srcH, size, scale, newW, newH, (size - newW) / 2, (size - newH) / 2)
        }
    }
}
