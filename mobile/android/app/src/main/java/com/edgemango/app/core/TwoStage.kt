package com.edgemango.app.core

data class TwoStageConfig(
    /** Stage-1 score a mango box needs to be kept (the mango-vs-background gate). */
    val mangoThreshold: Float = 0.5f,
    /** At most this many mangos per frame, best score first. */
    val maxMangos: Int = 10,
    /** Stage-2 damage confidence threshold. */
    val damageThreshold: Float = 0.25f,
    val computeMasks: Boolean = true,
    val computePolygons: Boolean = false,
    /** NMS IoU threshold for both stages; null keeps each sidecar's value. */
    val nmsIou: Float? = null,
    /** Stage 2 only: also suppress overlapping boxes of different classes. */
    val agnosticNms: Boolean = false,
    /** Crop margin around each mango box, as a fraction of its width/height ([RoiRule.MARGIN]). */
    val margin: Double = RoiRule.MARGIN,
)

/** One mango found by stage 1 and the damage found inside it by stage 2. */
class MangoDetection(
    val trackId: Int,
    val box: SourceBox,
    val score: Float,
    /** The stage-2 crop: the mango box plus margin, clipped to the frame. */
    val window: IntBox,
    /** Damage detections whose box centre lies inside [box], in frame pixels. */
    val damages: List<FrameDetection>,
    /** Damage masks of this mango on the prototype grid, or null (detect model, masks off, or no damage). */
    val maskLayer: MaskLayer?,
    val timings: FrameTimings,
) {
    val maxDamageScore: Float get() = damages.maxOfOrNull { it.score } ?: 0f
}

class TwoStageResult(
    val frameW: Int,
    val frameH: Int,
    /** Best score first. Empty means "no mango in the frame": stage 2 did not run. */
    val mangos: List<MangoDetection>,
    /** Time of the stage-1 pass alone. */
    val detectorMs: Double,
    /** Stage 1 plus every stage-2 pass, split by phase. */
    val timings: FrameTimings,
) {
    val noMango: Boolean get() = mangos.isEmpty()
    val maxDamageScore: Float get() = mangos.maxOfOrNull { it.maxDamageScore } ?: 0f
}

/**
 * Stage 1 finds mangos on the full frame; stage 2 (the damage model) runs on each mango's crop. When stage 1 finds no
 * mango above the threshold, stage 2 does not run, which is what makes the damage model's "mango" class unnecessary.
 * Mangos carry stable track ids from a [MangoTracker]. Not thread-safe (both pipelines reuse their buffers).
 */
class TwoStagePipeline(
    val detector: FramePipeline,
    val damage: FramePipeline,
    val tracker: MangoTracker = MangoTracker(),
) {
    fun process(frame: RgbImage, config: TwoStageConfig): TwoStageResult {
        val stage1 = detector.process(
            frame,
            PipelineConfig(RoiMode.FULL_FRAME, config.mangoThreshold, computeMasks = false, nmsIou = config.nmsIou),
        )
        val boxes = stage1.detections
            .filter { it.cls == FrameResult.MANGO_CLASS && it.score >= config.mangoThreshold }
            .sortedByDescending { it.score }
            .take(config.maxMangos.coerceAtLeast(0))
        // Always update, also with no boxes, so tracks age out.
        val ids = tracker.update(boxes.map { it.box })

        var roi = stage1.timings.roiMs
        var pre = stage1.timings.preMs
        var infer = stage1.timings.inferMs
        var post = stage1.timings.postMs
        val damageConfig = PipelineConfig(
            RoiMode.FULL_FRAME, config.damageThreshold, config.computeMasks, config.computePolygons,
            nmsIou = config.nmsIou, agnosticNms = config.agnosticNms,
        )
        val mangos = boxes.mapIndexed { i, m ->
            val b = m.box
            val window = RoiRule.window(b.x1, b.y1, b.x2, b.y2, frame.width, frame.height, config.margin)
                ?: RoiRule.full(frame.width, frame.height)
            val prepared = damage.prepareWindow(frame, window)
            val r = damage.processPrepared(frame, prepared, damageConfig) { d ->
                d.cls == FrameResult.DAMAGE_CLASS && b.contains(d.box.centerX, d.box.centerY)
            }
            roi += r.timings.roiMs
            pre += r.timings.preMs
            infer += r.timings.inferMs
            post += r.timings.postMs
            MangoDetection(ids[i], b, m.score, window, r.detections, r.maskLayer, r.timings)
        }
        return TwoStageResult(
            frame.width, frame.height, mangos, stage1.timings.totalMs, FrameTimings(roi, pre, infer, post),
        )
    }

    fun resetTracking() = tracker.reset()
}
