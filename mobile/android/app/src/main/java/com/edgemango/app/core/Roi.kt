package com.edgemango.app.core

import kotlin.math.ceil
import kotlin.math.floor

/** How the region fed to the model is chosen. */
enum class RoiMode(val label: String) {
    COLOR_MASK("Color mask"),
    FULL_FRAME("Full frame"),
    CENTER_SQUARE("Center square"),

    /** Two stages: a mango detector on the full frame, then the damage model on each mango's crop ([TwoStagePipeline]). */
    DETECTOR("Detector (2-stage)"),
}

/** The crop rules of `src/training/roi_crop.py`, applied to a mango box. */
object RoiRule {
    const val MARGIN = 0.10
    private const val PIXEL_EPS = 1e-3

    /**
     * Mango box plus `margin` of its width/height on each side, snapped outward to integers and clipped to
     * the frame (not squared). Null when the window would be empty, in which case training falls back to the
     * full frame.
     */
    fun window(
        x1: Double, y1: Double, x2: Double, y2: Double,
        frameW: Int, frameH: Int,
        margin: Double = MARGIN,
    ): IntBox? {
        val w = x2 - x1
        val h = y2 - y1
        val xMin = floor(x1 - margin * w + PIXEL_EPS).toInt().coerceIn(0, frameW)
        val yMin = floor(y1 - margin * h + PIXEL_EPS).toInt().coerceIn(0, frameH)
        val xMax = ceil(x2 + margin * w - PIXEL_EPS).toInt().coerceIn(0, frameW)
        val yMax = ceil(y2 + margin * h - PIXEL_EPS).toInt().coerceIn(0, frameH)
        if (xMax <= xMin || yMax <= yMin) return null
        return IntBox(xMin, yMin, xMax, yMax)
    }

    fun full(frameW: Int, frameH: Int) = IntBox(0, 0, frameW, frameH)

    fun centerSquare(frameW: Int, frameH: Int): IntBox {
        val side = minOf(frameW, frameH)
        val x = (frameW - side) / 2
        val y = (frameH - side) / 2
        return IntBox(x, y, x + side, y + side)
    }

    /** A box found on a frame downscaled by integer factors, back in full-frame pixels. */
    fun scaleBox(box: IntBox, factorX: Int, factorY: Int) =
        IntBox(box.x1 * factorX, box.y1 * factorY, box.x2 * factorX, box.y2 * factorY)
}
