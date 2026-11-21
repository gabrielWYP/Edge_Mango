package com.edgemango.app.core

/**
 * The model's raw outputs, resolved by role. `coefs` and `protos` are null for detect models. For an Ultralytics export the
 * whole output tensor is in `boxes` and `cls` is empty.
 */
class RawOutputs(
    val boxes: FloatArray,
    val cls: FloatArray,
    val coefs: FloatArray?,
    val protos: FloatArray?,
)

/** Runs the network on one preprocessed NHWC float32 RGB tensor. */
interface Inferencer {
    fun infer(input: FloatArray): RawOutputs
}

data class PipelineConfig(
    val roiMode: RoiMode = RoiMode.COLOR_MASK,
    val confThreshold: Float = 0.25f,
    val computeMasks: Boolean = true,
    /** Also trace each damage detection's mask into a frame-pixel polygon (needs a segment model). */
    val computePolygons: Boolean = false,
)

data class FrameTimings(val roiMs: Double, val preMs: Double, val inferMs: Double, val postMs: Double) {
    val totalMs: Double get() = roiMs + preMs + inferMs + postMs
}

/**
 * A detection with its box in source-frame pixels. [polygon] is the simplified outline of the largest blob of its mask,
 * flat `x0, y0, x1, y1, ...` in frame pixels (null unless [PipelineConfig.computePolygons] and a damage detection).
 */
class FrameDetection(val cls: Int, val score: Float, val box: SourceBox, val polygon: DoubleArray? = null) {
    fun copy(polygon: DoubleArray? = this.polygon) = FrameDetection(cls, score, box, polygon)

    override fun toString() = "FrameDetection(cls=$cls, score=$score, box=$box)"
}

/** Maps prototype-grid pixels to frame pixels: `frameX = sx * u + tx`, `frameY = sy * v + ty`. */
data class Affine(val sx: Double, val sy: Double, val tx: Double, val ty: Double)

/** Translucent instance masks of one frame, painted on the prototype grid (`width x height`, ARGB). */
class MaskLayer(val width: Int, val height: Int, val argb: IntArray, val toFrame: Affine) {
    companion object {
        fun affine(g: LetterboxGeometry, window: IntBox, protoW: Int, protoH: Int, imageSize: Int): Affine {
            val cellX = imageSize.toDouble() / protoW
            val cellY = imageSize.toDouble() / protoH
            return Affine(
                sx = cellX / g.scale,
                sy = cellY / g.scale,
                tx = window.x1 - g.padX / g.scale,
                ty = window.y1 - g.padY / g.scale,
            )
        }
    }
}

class FrameResult(
    val frameW: Int,
    val frameH: Int,
    val window: IntBox,
    /** Mango box found by the colour mask, in frame pixels (null in the other ROI modes or when nothing was found). */
    val mangoBox: IntBox?,
    val detections: List<FrameDetection>,
    val maskLayer: MaskLayer?,
    val timings: FrameTimings,
) {
    val maxDamageScore: Float get() = detections.filter { it.cls == DAMAGE_CLASS }.maxOfOrNull { it.score } ?: 0f

    companion object {
        const val MANGO_CLASS = 0
        const val DAMAGE_CLASS = 1
    }
}

/** What preprocessing decided for one frame; the model input itself lives in the pipeline's buffer. */
class Prepared(val window: IntBox, val geometry: LetterboxGeometry, val mangoBox: IntBox?, val roiMs: Double, val preMs: Double)

/**
 * ROI selection -> crop -> letterbox -> normalize -> inference -> decode -> masks, for one frame at a time.
 * Not thread-safe: it reuses its tensor buffers, so call it from one thread.
 */
class FramePipeline(
    val sidecar: ModelSidecar,
    private val inferencer: Inferencer,
    private val clockMs: () -> Double = { System.nanoTime() / 1e6 },
) {
    private val size = sidecar.inputSize
    val input = FloatArray(size * size * 3)
    private val canvas = RgbImage(size, size, IntArray(size * size))

    fun prepare(frame: RgbImage, config: PipelineConfig): Prepared {
        val t0 = clockMs()
        var mangoBox: IntBox? = null
        val window: IntBox = when (config.roiMode) {
            // DETECTOR is driven by TwoStagePipeline, which supplies each window through prepareWindow.
            RoiMode.FULL_FRAME, RoiMode.DETECTOR -> RoiRule.full(frame.width, frame.height)
            RoiMode.CENTER_SQUARE -> RoiRule.centerSquare(frame.width, frame.height)
            RoiMode.COLOR_MASK -> {
                val f = MangoColorSegmenter.downscaleFactor(frame.width)
                val small = MangoColorSegmenter.downscale(frame, f)
                val box = MangoColorSegmenter.segment(small).bbox()?.let { RoiRule.scaleBox(it, f, f) }
                mangoBox = box
                box?.let { RoiRule.window(it.x1.toDouble(), it.y1.toDouble(), it.x2.toDouble(), it.y2.toDouble(), frame.width, frame.height) }
                    ?: RoiRule.full(frame.width, frame.height)
            }
        }
        val t1 = clockMs()
        return prepareWindow(frame, window, mangoBox, t1 - t0)
    }

    /** Crop [window] of [frame], letterbox it and fill [input]; the ROI step already happened, taking [roiMs]. */
    fun prepareWindow(frame: RgbImage, window: IntBox, mangoBox: IntBox? = null, roiMs: Double = 0.0): Prepared {
        val t1 = clockMs()
        val g = LetterboxGeometry.of(window.width, window.height, size)
        Letterbox.toCanvas(frame, window, g, sidecar.padValue, canvas)
        Letterbox.normalizeInto(canvas, input, sidecar.normalization)
        val t2 = clockMs()
        return Prepared(window, g, mangoBox, roiMs, t2 - t1)
    }

    fun process(frame: RgbImage, config: PipelineConfig): FrameResult = processPrepared(frame, prepare(frame, config), config)

    /** Inference + post-processing of an already [prepared] frame; [accept] drops detections before masks are built. */
    fun processPrepared(
        frame: RgbImage,
        prepared: Prepared,
        config: PipelineConfig,
        accept: ((FrameDetection) -> Boolean)? = null,
    ): FrameResult {
        val t0 = clockMs()
        val raw = inferencer.infer(input)
        val t1 = clockMs()
        val (detections, layer) = postprocess(raw, prepared, config, accept)
        val t2 = clockMs()
        return FrameResult(
            frame.width, frame.height, prepared.window, prepared.mangoBox, detections, layer,
            FrameTimings(prepared.roiMs, prepared.preMs, t1 - t0, t2 - t1),
        )
    }

    /** Decode + masks of [raw], which the model produced for [prepared]'s input. */
    fun postprocess(
        raw: RawOutputs,
        prepared: Prepared,
        config: PipelineConfig,
        accept: ((FrameDetection) -> Boolean)? = null,
    ): Pair<List<FrameDetection>, MaskLayer?> {
        val params = sidecar.decodeParams().copy(confThreshold = config.confThreshold)
        var dets = if (sidecar.isUltralytics) {
            UltralyticsDecoder.decode(raw.boxes, params, sidecar.outputLayout)
        } else {
            Decoder.decode(raw.boxes, raw.cls, params)
        }
        val w = prepared.window
        val g = prepared.geometry
        var out = dets.map { FrameDetection(it.cls, it.score, it.toSourceBox(g, w.x1, w.y1, w.width, w.height)) }
        if (accept != null) {
            val keep = out.indices.filter { accept(out[it]) }
            dets = keep.map { dets[it] }
            out = keep.map { out[it] }
        }
        val wantMasks = config.computeMasks || config.computePolygons
        if (!wantMasks || raw.coefs == null || raw.protos == null || sidecar.task != ModelTask.SEGMENT) return out to null

        val nm = sidecar.numPrototypes
        val cells = raw.protos.size / nm
        val side = Math.round(Math.sqrt(cells.toDouble())).toInt()
        require(side * side == cells) { "prototypes are not square: ${raw.protos.size} values for $nm channels" }
        val masks = dets.map { MaskAssembler.assemble(it, raw.coefs, raw.protos, nm, side, side, size) }
        val affine = MaskLayer.affine(g, w, side, side, size)
        if (config.computePolygons) {
            out = out.mapIndexed { i, d ->
                if (d.cls != FrameResult.DAMAGE_CLASS) d
                else d.copy(polygon = MaskPolygon.largestContour(masks[i], side, side)?.let { toFrame(it, affine) })
            }
        }
        val layer = if (config.computeMasks) buildLayer(dets, masks, side, affine) else null
        return out to layer
    }

    private fun toFrame(gridXy: DoubleArray, a: Affine): DoubleArray =
        DoubleArray(gridXy.size) { i -> if (i % 2 == 0) a.sx * gridXy[i] + a.tx else a.sy * gridXy[i] + a.ty }

    private fun buildLayer(dets: List<Detection>, masks: List<BooleanArray>, side: Int, affine: Affine): MaskLayer {
        val argb = IntArray(side * side)
        // Mango first, damage on top.
        for (cls in 0 until sidecar.numClasses) {
            val color = MASK_COLORS[cls % MASK_COLORS.size]
            for ((i, d) in dets.withIndex()) {
                if (d.cls != cls) continue
                val mask = masks[i]
                for (j in mask.indices) if (mask[j]) argb[j] = color
            }
        }
        return MaskLayer(side, side, argb, affine)
    }

    companion object {
        /** Translucent ARGB: mango green, damage red. */
        val MASK_COLORS = intArrayOf(0x6600C853, 0x88FF1744.toInt())
    }
}
