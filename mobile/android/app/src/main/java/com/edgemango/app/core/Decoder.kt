package com.edgemango.app.core

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

data class DecodeParams(
    val confThreshold: Float = 0.25f,
    val nmsIouThreshold: Float = 0.5f,
    val maxDetections: Int = 300,
    val imageSize: Int = 640,
    val strides: List<Int> = listOf(8, 16, 32),
    val numClasses: Int = 2,
    /**
     * Safety cap on candidates entering NMS (highest scores first). A trained model never gets near it; it only
     * keeps a degenerate (e.g. random-init) model from stalling the frame loop. 0 disables it.
     */
    val maxCandidates: Int = 3000,
    /** Suppress overlapping boxes of different classes too (one NMS over all classes). */
    val agnosticNms: Boolean = false,
)

/** One detection in letterboxed-input pixels (`cx, cy, w, h`) and the anchor it came from. */
data class Detection(
    val cls: Int,
    val score: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val anchor: Int,
) {
    /** The box in the frame the crop was taken from: undo the letterbox, clip to the crop, add the crop origin. */
    fun toSourceBox(g: LetterboxGeometry, cropX: Int, cropY: Int, cropW: Int, cropH: Int): SourceBox {
        fun mapX(v: Float) = ((v - g.padX) / g.scale).coerceIn(0.0, cropW.toDouble()) + cropX
        fun mapY(v: Float) = ((v - g.padY) / g.scale).coerceIn(0.0, cropH.toDouble()) + cropY
        return SourceBox(mapX(cx - w / 2), mapY(cy - h / 2), mapX(cx + w / 2), mapY(cy + h / 2))
    }
}

/**
 * Decode of the RAW exported outputs, mirroring `src/export/reference.py::decode_exported_outputs`
 * (which delegates to `src/training/decode.py`):
 *
 * - anchors are level-major (finest stride first), row-major inside a level;
 * - `cx = (gx + 0.5) * stride + dx`, `cy = (gy + 0.5) * stride + dy`, `w = exp(log_w)`, `h = exp(log_h)`
 *   (`dx, dy` are already pixels, not multiplied by the stride);
 * - `score = sigmoid(logit)`, one candidate per (anchor, class) with `score > conf` (not argmax);
 * - greedy per-class NMS (suppress when IoU > threshold), highest score first; at most `maxDetections`.
 */
object Decoder {
    fun decode(boxesRaw: FloatArray, clsLogits: FloatArray, params: DecodeParams = DecodeParams()): List<Detection> {
        val nc = params.numClasses
        val anchors = params.strides.sumOf { (params.imageSize / it) * (params.imageSize / it) }
        require(boxesRaw.size >= anchors * 4) { "boxes_raw has ${boxesRaw.size} values, need ${anchors * 4}" }
        require(clsLogits.size >= anchors * nc) { "cls_logits has ${clsLogits.size} values, need ${anchors * nc}" }

        val conf = params.confThreshold
        // sigmoid(x) > conf  <=>  x > logit(conf); a small guard band keeps the exact sigmoid comparison authoritative.
        val logitFloor = if (conf <= 0f) Float.NEGATIVE_INFINITY else if (conf >= 1f) Float.POSITIVE_INFINITY else ln(conf / (1f - conf)) - 0.01f

        val cand = ArrayList<Detection>(256)
        var offset = 0
        for (stride in params.strides) {
            val grid = params.imageSize / stride
            for (a in 0 until grid * grid) {
                val i = offset + a
                for (c in 0 until nc) {
                    val logit = clsLogits[i * nc + c]
                    if (logit < logitFloor) continue
                    val score = sigmoid(logit)
                    if (score > conf) {
                        val gy = a / grid
                        val gx = a % grid
                        val dx = boxesRaw[i * 4]
                        val dy = boxesRaw[i * 4 + 1]
                        val w = exp(boxesRaw[i * 4 + 2].toDouble()).toFloat()
                        val h = exp(boxesRaw[i * 4 + 3].toDouble()).toFloat()
                        cand.add(Detection(c, score, (gx + 0.5f) * stride + dx, (gy + 0.5f) * stride + dy, w, h, i))
                    }
                }
            }
            offset += grid * grid
        }
        if (cand.isEmpty()) return emptyList()
        cand.sortByDescending { it.score }
        val pool = if (params.maxCandidates > 0 && cand.size > params.maxCandidates) cand.subList(0, params.maxCandidates) else cand
        return nms(pool, params)
    }

    private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

    /**
     * Greedy NMS over a score-sorted list (suppress when IoU > threshold), per class unless [DecodeParams.agnosticNms];
     * the result is score-sorted and capped. Shared by every decoder.
     */
    fun nms(sorted: List<Detection>, params: DecodeParams): List<Detection> {
        val kept = ArrayList<Detection>()
        val groups: List<List<Detection>> =
            if (params.agnosticNms) listOf(sorted) else (0 until params.numClasses).map { c -> sorted.filter { it.cls == c } }
        for (items in groups) {
            if (items.isEmpty()) continue
            val suppressed = BooleanArray(items.size)
            var keptInGroup = 0
            for (i in items.indices) {
                if (suppressed[i]) continue
                kept.add(items[i])
                if (++keptInGroup >= params.maxDetections) break
                for (j in i + 1 until items.size) {
                    if (!suppressed[j] && iou(items[i], items[j]) > params.nmsIouThreshold) suppressed[j] = true
                }
            }
        }
        kept.sortByDescending { it.score }
        return if (kept.size > params.maxDetections) kept.subList(0, params.maxDetections).toList() else kept
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ax1 = a.cx - a.w / 2
        val ay1 = a.cy - a.h / 2
        val ax2 = a.cx + a.w / 2
        val ay2 = a.cy + a.h / 2
        val bx1 = b.cx - b.w / 2
        val by1 = b.cy - b.h / 2
        val bx2 = b.cx + b.w / 2
        val by2 = b.cy + b.h / 2
        val iw = max(0f, min(ax2, bx2) - max(ax1, bx1))
        val ih = max(0f, min(ay2, by2) - max(ay1, by1))
        val inter = iw * ih
        val union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return if (union <= 0f) 0f else inter / union
    }
}

/** `mask = sigmoid(coefs . protos)` cropped to the detection box in prototype coordinates (`src/training/decode.py::process_masks`). */
object MaskAssembler {
    /**
     * @param coefs all anchors' coefficients, `anchors x nm`, row-major (the row of `det.anchor` is used)
     * @param protos `nm x ph x pw`, NCHW
     * @return `ph * pw` booleans, row-major, at prototype resolution (no upsampling)
     */
    fun assemble(det: Detection, coefs: FloatArray, protos: FloatArray, nm: Int, ph: Int, pw: Int, imageSize: Int): BooleanArray {
        val out = BooleanArray(ph * pw)
        // Normalised box exactly as the float32 reference computes it.
        val cx = det.cx / imageSize
        val cy = det.cy / imageSize
        val bw = det.w / imageSize
        val bh = det.h / imageSize
        val fx1 = (cx - bw / 2) * pw
        val fx2 = (cx + bw / 2) * pw
        val fy1 = (cy - bh / 2) * ph
        val fy2 = (cy + bh / 2) * ph
        // Integer pixel i is inside when fx1 <= i < fx2.
        val c0 = max(0, Math.ceil(fx1.toDouble()).toInt())
        val c1 = min(pw, Math.ceil(fx2.toDouble()).toInt())
        val r0 = max(0, Math.ceil(fy1.toDouble()).toInt())
        val r1 = min(ph, Math.ceil(fy2.toDouble()).toInt())
        if (c1 <= c0 || r1 <= r0) return out

        val width = c1 - c0
        val acc = FloatArray(width)
        val base = det.anchor * nm
        for (r in r0 until r1) {
            java.util.Arrays.fill(acc, 0f)
            for (k in 0 until nm) {
                val coef = coefs[base + k]
                if (coef == 0f) continue
                val p = k * ph * pw + r * pw + c0
                for (x in 0 until width) acc[x] += coef * protos[p + x]
            }
            for (x in 0 until width) {
                val v = acc[x]
                // sigmoid(v) > 0.5; evaluate the sigmoid only where float rounding could matter.
                out[r * pw + c0 + x] = if (v > 1e-6f) true else if (v < -1e-6f) false else (1.0 / (1.0 + exp(-v.toDouble()))).toFloat() > 0.5f
            }
        }
        return out
    }
}
