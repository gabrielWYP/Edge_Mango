package com.edgemango.app.ui.live

import android.graphics.Bitmap
import android.graphics.DashPathEffect
import android.graphics.Matrix
import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.edgemango.app.core.FrameResult
import com.edgemango.app.core.IntBox
import com.edgemango.app.core.MaskLayer
import com.edgemango.app.core.RoiMode
import com.edgemango.app.core.SourceBox
import com.edgemango.app.core.TwoStageResult
import com.edgemango.app.data.DisplayMode

private const val MANGO_COLOR = 0xFF00E676.toInt()
private const val DAMAGE_COLOR = 0xFFFF1744.toInt()
private const val ROI_COLOR = 0xFFFFEA00.toInt()
private const val MASK_BOX_COLOR = 0xFF00B0FF.toInt()

/**
 * Draws one frame's result over the preview. Stateless: everything comes from [result] (single-crop ROI modes) or
 * [twoStage] (DETECTOR mode: every mango box with its track id, and the damage boxes/masks inside it).
 *
 * The preview uses FIT_CENTER, so the upright frame (`frameW x frameH`) maps onto the view with a uniform scale
 * and a centring offset; boxes, the ROI rectangle and the mask bitmaps all go through that same transform.
 */
@Composable
fun DetectionOverlay(
    result: FrameResult?,
    twoStage: TwoStageResult?,
    displayMode: DisplayMode,
    roiMode: RoiMode,
    modifier: Modifier = Modifier,
) {
    val density = LocalDensity.current
    val stroke = with(density) { 2.5.dp.toPx() }
    val textSize = with(density) { 13.sp.toPx() }
    val paints = remember(stroke, textSize) { OverlayPaints(stroke, textSize) }
    val maskBitmap = remember(result) {
        if (result?.maskLayer != null && displayMode.showMask) toBitmap(result.maskLayer) else null
    }
    val mangoBitmaps = remember(twoStage) {
        twoStage?.mangos?.map { m -> if (m.maskLayer != null && displayMode.showMask) toBitmap(m.maskLayer) else null }
    }

    Canvas(modifier) {
        val frameW = result?.frameW ?: twoStage?.frameW ?: return@Canvas
        val frameH = result?.frameH ?: twoStage?.frameH ?: return@Canvas
        val scale = minOf(size.width / frameW, size.height / frameH)
        val offX = (size.width - frameW * scale) / 2f
        val offY = (size.height - frameH * scale) / 2f
        val t = Transform(scale, offX, offY)
        drawIntoCanvas { canvas ->
            val c = canvas.nativeCanvas
            if (result != null) drawSingle(c, result, maskBitmap, displayMode, roiMode, t, paints, textSize)
            if (twoStage != null) drawTwoStage(c, twoStage, mangoBitmaps.orEmpty(), displayMode, t, paints, textSize)
        }
    }
}

private class Transform(val scale: Float, val offX: Float, val offY: Float) {
    fun x(v: Double) = offX + v.toFloat() * scale
    fun y(v: Double) = offY + v.toFloat() * scale
}

private fun toBitmap(m: MaskLayer): Bitmap = Bitmap.createBitmap(m.argb, m.width, m.height, Bitmap.Config.ARGB_8888)

private fun drawMask(c: android.graphics.Canvas, bitmap: Bitmap, layer: MaskLayer, window: IntBox, t: Transform, paint: Paint) {
    val a = layer.toFrame
    val m = Matrix()
    m.setValues(
        floatArrayOf(
            (a.sx * t.scale).toFloat(), 0f, (a.tx * t.scale + t.offX).toFloat(),
            0f, (a.sy * t.scale).toFloat(), (a.ty * t.scale + t.offY).toFloat(),
            0f, 0f, 1f,
        ),
    )
    c.save()
    c.clipRect(t.x(window.x1.toDouble()), t.y(window.y1.toDouble()), t.x(window.x2.toDouble()), t.y(window.y2.toDouble()))
    c.drawBitmap(bitmap, m, paint)
    c.restore()
}

private fun drawBox(c: android.graphics.Canvas, b: SourceBox, label: String, paint: ClassPaints, t: Transform, textSize: Float) {
    val x1 = t.x(b.x1)
    val y1 = t.y(b.y1)
    c.drawRect(x1, y1, t.x(b.x2), t.y(b.y2), paint.box)
    c.drawText(label, x1 + 4f, maxOf(y1 - 6f, textSize), paint.text)
}

private fun drawSingle(
    c: android.graphics.Canvas, r: FrameResult, maskBitmap: Bitmap?, displayMode: DisplayMode, roiMode: RoiMode,
    t: Transform, paints: OverlayPaints, textSize: Float,
) {
    val w = r.window
    // ROI window and, for the colour mode, the raw mango box it came from.
    c.drawRect(t.x(w.x1.toDouble()), t.y(w.y1.toDouble()), t.x(w.x2.toDouble()), t.y(w.y2.toDouble()), paints.roi)
    r.mangoBox?.takeIf { roiMode == RoiMode.COLOR_MASK }?.let { b ->
        c.drawRect(t.x(b.x1.toDouble()), t.y(b.y1.toDouble()), t.x(b.x2.toDouble()), t.y(b.y2.toDouble()), paints.maskBox)
    }
    val layer = r.maskLayer
    if (maskBitmap != null && layer != null) drawMask(c, maskBitmap, layer, w, t, paints.mask)
    if (displayMode.showBoxes) {
        for (d in r.detections.sortedBy { it.cls }) {
            val isDamage = d.cls == FrameResult.DAMAGE_CLASS
            val label = (if (isDamage) "damage " else "mango ") + "%.2f".format(d.score)
            drawBox(c, d.box, label, if (isDamage) paints.damage else paints.mango, t, textSize)
        }
    }
}

private fun drawTwoStage(
    c: android.graphics.Canvas, r: TwoStageResult, bitmaps: List<Bitmap?>, displayMode: DisplayMode,
    t: Transform, paints: OverlayPaints, textSize: Float,
) {
    for ((i, m) in r.mangos.withIndex()) {
        val layer = m.maskLayer
        val bmp = bitmaps.getOrNull(i)
        if (bmp != null && layer != null) drawMask(c, bmp, layer, m.window, t, paints.mask)
        if (displayMode.showBoxes) {
            val label = "#${m.trackId} %.2f".format(m.score) + if (m.held) " (held)" else ""
            drawBox(c, m.box, label, if (m.held) paints.heldMango else paints.mango, t, textSize)
            for (d in m.damages) drawBox(c, d.box, "%.2f".format(d.score), paints.damage, t, textSize)
        }
    }
}

private class ClassPaints(color: Int, stroke: Float, textSize: Float) {
    val box = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = stroke; this.color = color }
    val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { this.color = color; this.textSize = textSize; setShadowLayer(3f, 0f, 0f, 0xFF000000.toInt()) }
}

private class OverlayPaints(stroke: Float, textSize: Float) {
    val mango = ClassPaints(MANGO_COLOR, stroke, textSize)
    val damage = ClassPaints(DAMAGE_COLOR, stroke, textSize)
    /** Dimmer, dashed variant for mangos held from tracker memory. */
    val heldMango = ClassPaints(MANGO_COLOR, stroke, textSize).also {
        it.box.alpha = 140
        it.box.pathEffect = DashPathEffect(floatArrayOf(stroke * 3, stroke * 2), 0f)
        it.text.alpha = 160
    }
    val roi = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = stroke; color = ROI_COLOR
        pathEffect = DashPathEffect(floatArrayOf(stroke * 4, stroke * 3), 0f)
    }
    val maskBox = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = stroke / 2; color = MASK_BOX_COLOR }
    val mask = Paint(Paint.FILTER_BITMAP_FLAG)
}
