package com.edgemango.app.core

/** One live track as reported by [MangoTracker.update]: its box and score are the last seen when [held]. */
class TrackedBox(val id: Int, val box: SourceBox, val score: Float, val held: Boolean)

/**
 * Greedy IoU tracker that gives every mango a stable id across frames.
 *
 * Each [update] matches the frame's boxes against the last known box of every live track, highest IoU first, accepting a
 * pair only when `IoU >= iouThreshold`. Unmatched boxes start new tracks; a track not matched for more than [maxAge]
 * consecutive frames is dropped (its id is never reused). Pure Kotlin, one instance per camera stream, not thread-safe.
 *
 * The scored [update] adds hysteresis and hold: a candidate that matches a live track survives down to `keepThreshold`,
 * a new track needs `startThreshold`, and a track missed for at most `holdFrames` frames is still reported (held).
 */
class MangoTracker(private val iouThreshold: Double = 0.3, private val maxAge: Int = 15) {
    private class Track(val id: Int, var box: SourceBox, var score: Float, var missed: Int = 0)

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    /** Tracks currently alive (seen within the last [maxAge] frames). */
    val activeTracks: Int get() = tracks.size

    /** Track ids of [boxes], in the same order (no hysteresis, no hold). */
    fun update(boxes: List<SourceBox>): IntArray {
        val out = update(boxes.map { it to 1f }, 0f, 0f, 0)
        return IntArray(boxes.size) { out[it].id }
    }

    /**
     * Scored update. Returns the accepted candidates first (input order, [TrackedBox.held] false), then held tracks
     * (held true, last box and last score). A candidate below `keepThreshold`, or below `startThreshold` without a
     * matching live track, is discarded. `holdFrames` is clamped to [maxAge].
     */
    fun update(
        candidates: List<Pair<SourceBox, Float>>,
        startThreshold: Float,
        keepThreshold: Float,
        holdFrames: Int,
    ): List<TrackedBox> {
        val hold = holdFrames.coerceIn(0, maxAge)
        val matched = arrayOfNulls<Track>(candidates.size) // matched live track per candidate
        val pairs = ArrayList<Triple<Double, Int, Int>>() // iou, track index, candidate index
        for ((ti, t) in tracks.withIndex()) for ((ci, c) in candidates.withIndex()) {
            if (c.second < keepThreshold) continue
            val iou = t.box.iou(c.first)
            if (iou >= iouThreshold) pairs.add(Triple(iou, ti, ci))
        }
        pairs.sortWith(compareByDescending<Triple<Double, Int, Int>> { it.first }.thenBy { it.second }.thenBy { it.third })
        val trackUsed = BooleanArray(tracks.size)
        for ((_, ti, ci) in pairs) {
            if (trackUsed[ti] || matched[ci] != null) continue
            val t = tracks[ti]
            trackUsed[ti] = true
            matched[ci] = t
            t.box = candidates[ci].first
            t.score = candidates[ci].second
            t.missed = 0
        }
        val held = ArrayList<TrackedBox>()
        for ((ti, t) in tracks.withIndex()) if (!trackUsed[ti]) {
            t.missed++
            if (t.missed <= hold) held.add(TrackedBox(t.id, t.box, t.score, true))
        }
        tracks.removeAll { it.missed > maxAge }
        val out = ArrayList<TrackedBox>()
        for ((ci, c) in candidates.withIndex()) {
            val t = matched[ci]
            if (t != null) {
                out.add(TrackedBox(t.id, c.first, c.second, false))
            } else if (c.second >= startThreshold && c.second >= keepThreshold) {
                val nt = Track(nextId++, c.first, c.second)
                tracks.add(nt)
                out.add(TrackedBox(nt.id, c.first, c.second, false))
            }
        }
        return out + held
    }

    /** Forgets every track; ids start again at 1. */
    fun reset() {
        tracks.clear()
        nextId = 1
    }
}
