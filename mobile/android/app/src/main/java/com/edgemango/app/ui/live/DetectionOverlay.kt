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
import com.edgemango.app.data.DisplayMode
import com.edgemango.app.core.RoiMode

private const val MANGO_COLOR = 0xFF00E676.toInt()
private const val DAMAGE_COLOR = 0xFFFF1744.toInt()
private const val ROI_COLOR = 0xFFFFEA00.toInt()
private const val MASK_BOX_COLOR = 0xFF00B0FF.toInt()

/**
 * Draws one frame's result over the preview. Stateless: everything comes from [result].
 *
 * The preview uses FIT_CENTER, so the upright frame (`result.frameW x frameH`) maps onto the view with a uniform scale
 * and a centring offset; boxes, the ROI rectangle and the mask bitmap all go through that same transform.
 */
@Composable
fun DetectionOverlay(
    result: FrameResult?,
    displayMode: DisplayMode,
    roiMode: RoiMode,
    modifier: Modifier = Modifier,
) {
    val density = LocalDensity.current
    val stroke = with(density) { 2.5.dp.toPx() }
    val textSize = with(density) { 13.sp.toPx() }
    val paints = remember(stroke, textSize) { OverlayPaints(stroke, textSize) }
    val maskBitmap = remember(result) {
        if (result?.maskLayer != null && displayMode.showMask) {
            val m = result.maskLayer
            Bitmap.createBitmap(m.argb, m.width, m.height, Bitmap.Config.ARGB_8888)
        } else null
    }

    Canvas(modifier) {
        val r = result ?: return@Canvas
        val scale = minOf(size.width / r.frameW, size.height / r.frameH)
        val offX = (size.width - r.frameW * scale) / 2f
        val offY = (size.height - r.frameH * scale) / 2f
        drawIntoCanvas { canvas ->
            val c = canvas.nativeCanvas
            val w = r.window
            // ROI window and, for the colour mode, the raw mango box it came from.
            c.drawRect(offX + w.x1 * scale, offY + w.y1 * scale, offX + w.x2 * scale, offY + w.y2 * scale, paints.roi)
            r.mangoBox?.takeIf { roiMode == RoiMode.COLOR_MASK }?.let { b ->
                c.drawRect(offX + b.x1 * scale, offY + b.y1 * scale, offX + b.x2 * scale, offY + b.y2 * scale, paints.maskBox)
            }

            val layer = r.maskLayer
            if (maskBitmap != null && layer != null) {
                val a = layer.toFrame
                val m = Matrix()
                m.setValues(
                    floatArrayOf(
                        (a.sx * scale).toFloat(), 0f, (a.tx * scale + offX).toFloat(),
                        0f, (a.sy * scale).toFloat(), (a.ty * scale + offY).toFloat(),
                        0f, 0f, 1f,
                    ),
                )
                c.save()
                c.clipRect(offX + w.x1 * scale, offY + w.y1 * scale, offX + w.x2 * scale, offY + w.y2 * scale)
                c.drawBitmap(maskBitmap, m, paints.mask)
                c.restore()
            }

            if (displayMode.showBoxes) {
                for (d in r.detections.sortedBy { it.cls }) {
                    val paint = if (d.cls == FrameResult.DAMAGE_CLASS) paints.damage else paints.mango
                    val x1 = offX + d.box.x1.toFloat() * scale
                    val y1 = offY + d.box.y1.toFloat() * scale
                    c.drawRect(x1, y1, offX + d.box.x2.toFloat() * scale, offY + d.box.y2.toFloat() * scale, paint.box)
                    val label = (if (d.cls == FrameResult.DAMAGE_CLASS) "damage " else "mango ") + "%.2f".format(d.score)
                    c.drawText(label, x1 + 4f, maxOf(y1 - 6f, textSize), paint.text)
                }
            }
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
    val roi = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = stroke; color = ROI_COLOR
        pathEffect = DashPathEffect(floatArrayOf(stroke * 4, stroke * 3), 0f)
    }
    val maskBox = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = stroke / 2; color = MASK_BOX_COLOR }
    val mask = Paint(Paint.FILTER_BITMAP_FLAG)
}
