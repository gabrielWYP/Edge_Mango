package com.edgemango.app.core

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlin.math.round

/**
 * File formats of a field-test session (`files/results/session_<stamp>/`), all pure Kotlin so they are JVM-tested:
 *
 * - `predictions.jsonl`: one [FrameRecord] per processed frame, boxes `[x1, y1, x2, y2]` in upright frame pixels;
 * - `session.json`: one [SessionInfo].
 *
 * `scripts/field_eval.py` reads exactly these fields.
 */
@Serializable
data class DetectionRecord(val cls: Int, val box: List<Double>, val score: Double)

@Serializable
data class DamageRecord(
    val box: List<Double>,
    val score: Double,
    /** Outline of the largest blob of the damage mask, flat `x0, y0, x1, y1, ...` in frame pixels (segment models only). */
    val polygon: List<Double>? = null,
)

@Serializable
data class MangoRecord(
    val trackId: Int,
    val box: List<Double>,
    val score: Double,
    val damages: List<DamageRecord>,
    val maxDamage: Double,
    /** True when the box is the tracker's memory of an earlier frame (stage 1 missed it here), not a detection of this frame. */
    val held: Boolean = false,
)

@Serializable
data class TimingsRecord(
    val roiMs: Double,
    val preMs: Double,
    val inferMs: Double,
    val postMs: Double,
    val totalMs: Double,
    /** Stage-1 share of the total, in DETECTOR mode only. */
    val detectorMs: Double? = null,
)

@Serializable
data class FrameRecord(
    val index: Int,
    val wallMs: Long,
    /** `ImageProxy.imageInfo.timestamp`: the camera's monotonic sensor timestamp. */
    val cameraTimestampNs: Long,
    val frameW: Int,
    val frameH: Int,
    val roiMode: String,
    val confThreshold: Double,
    val mangoThreshold: Double,
    val nmsIou: Double,
    val agnosticNms: Boolean,
    val maxMangos: Int,
    /** Path of the saved JPEG relative to the session folder; null when the write queue was full and the frame was dropped. */
    val frame: String?,
    val accelerator: String,
    val detectorModel: String?,
    val damageModel: String,
    /** DETECTOR mode: stage 1 found no mango, so stage 2 did not run. Null in the single-crop modes. */
    val noMango: Boolean? = null,
    /** DETECTOR mode: one entry per mango. Empty in the single-crop modes. */
    val mangos: List<MangoRecord> = emptyList(),
    /** Single-crop modes: every detection of the one crop (null in DETECTOR mode). */
    val detections: List<DetectionRecord>? = null,
    val timings: TimingsRecord,
    /** Stage-1 score a box matching a live track needs (hysteresis); `mangoThreshold` starts a track. */
    val keepThreshold: Double = 0.0,
    /** Frames a missed track is still reported (`held`). */
    val holdFrames: Int = 0,
)

/** Settings that shaped a frame record; copied into every line so a line is self-describing. */
data class FrameContext(
    val roiMode: String,
    val confThreshold: Float,
    val mangoThreshold: Float,
    val nmsIou: Float,
    val agnosticNms: Boolean,
    val maxMangos: Int,
    val accelerator: String,
    val detectorModel: String?,
    val damageModel: String,
    val keepThreshold: Float = 0f,
    val holdFrames: Int = 0,
)

@Serializable
data class ModelInfo(val name: String, val accelerator: String? = null, val sidecar: JsonElement? = null)

@Serializable
data class SessionSettings(
    val roiMode: String,
    val displayMode: String,
    val confidence: Double,
    val mangoThreshold: Double,
    val maxMangos: Int,
    val nmsIou: Double,
    val agnosticNms: Boolean,
    val analysisWidth: Int,
    val acceleratorChoice: String,
    val keepThreshold: Double = 0.0,
    val holdFrames: Int = 0,
)

@Serializable
data class SessionInfo(
    val session: String,
    val appVersion: String,
    val device: String,
    val soc: String,
    val android: String,
    val startedAtMs: Long,
    val endedAtMs: Long,
    val frameCount: Int,
    val framesSaved: Int,
    val droppedWrites: Int,
    val failedWrites: Int,
    val settings: SessionSettings,
    val damageModel: ModelInfo,
    val detectorModel: ModelInfo? = null,
)

object SessionJson {
    private val json = Json { encodeDefaults = true; explicitNulls = false }
    private val pretty = Json { encodeDefaults = true; explicitNulls = false; prettyPrint = true }

    /** One `predictions.jsonl` line (no trailing newline). */
    fun frameLine(r: FrameRecord): String = json.encodeToString(FrameRecord.serializer(), r)

    fun parseFrame(line: String): FrameRecord = json.decodeFromString(FrameRecord.serializer(), line)

    fun sessionJson(info: SessionInfo): String = pretty.encodeToString(SessionInfo.serializer(), info)

    fun parseSession(text: String): SessionInfo = json.decodeFromString(SessionInfo.serializer(), text)

    private fun r1(v: Double) = round(v * 10) / 10
    private fun r2(v: Double) = round(v * 100) / 100
    private fun r4(v: Double) = round(v * 10000) / 10000

    private fun boxOf(b: SourceBox) = listOf(r1(b.x1), r1(b.y1), r1(b.x2), r1(b.y2))
    private fun timings(t: FrameTimings, detectorMs: Double? = null) =
        TimingsRecord(r2(t.roiMs), r2(t.preMs), r2(t.inferMs), r2(t.postMs), r2(t.totalMs), detectorMs?.let(::r2))

    private fun damage(d: FrameDetection) =
        DamageRecord(boxOf(d.box), r4(d.score.toDouble()), d.polygon?.map(::r1))

    fun fromTwoStage(index: Int, wallMs: Long, cameraNs: Long, frameFile: String?, ctx: FrameContext, r: TwoStageResult) = FrameRecord(
        index, wallMs, cameraNs, r.frameW, r.frameH, ctx.roiMode,
        ctx.confThreshold.toDouble(), r4(ctx.mangoThreshold.toDouble()), ctx.nmsIou.toDouble(), ctx.agnosticNms, ctx.maxMangos,
        frameFile, ctx.accelerator, ctx.detectorModel, ctx.damageModel,
        noMango = r.noMango,
        mangos = r.mangos.map { m ->
            MangoRecord(m.trackId, boxOf(m.box), r4(m.score.toDouble()), m.damages.map(::damage), r4(m.maxDamageScore.toDouble()), m.held)
        },
        timings = timings(r.timings, r.detectorMs),
        keepThreshold = r4(ctx.keepThreshold.toDouble()), holdFrames = ctx.holdFrames,
    )

    fun fromSingleCrop(index: Int, wallMs: Long, cameraNs: Long, frameFile: String?, ctx: FrameContext, r: FrameResult) = FrameRecord(
        index, wallMs, cameraNs, r.frameW, r.frameH, ctx.roiMode,
        ctx.confThreshold.toDouble(), r4(ctx.mangoThreshold.toDouble()), ctx.nmsIou.toDouble(), ctx.agnosticNms, ctx.maxMangos,
        frameFile, ctx.accelerator, ctx.detectorModel, ctx.damageModel,
        detections = r.detections.map { DetectionRecord(it.cls, boxOf(it.box), r4(it.score.toDouble())) },
        timings = timings(r.timings),
        keepThreshold = r4(ctx.keepThreshold.toDouble()), holdFrames = ctx.holdFrames,
    )
}
