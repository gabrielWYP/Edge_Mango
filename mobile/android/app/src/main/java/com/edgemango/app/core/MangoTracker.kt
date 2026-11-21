package com.edgemango.app.core

/**
 * Greedy IoU tracker that gives every mango a stable id across frames.
 *
 * Each [update] matches the frame's boxes against the last known box of every live track, highest IoU first, accepting a
 * pair only when `IoU >= iouThreshold`. Unmatched boxes start new tracks; a track not matched for more than [maxAge]
 * consecutive frames is dropped (its id is never reused). Pure Kotlin, one instance per camera stream, not thread-safe.
 */
class MangoTracker(private val iouThreshold: Double = 0.3, private val maxAge: Int = 15) {
    private class Track(val id: Int, var box: SourceBox, var missed: Int = 0)

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    /** Tracks currently alive (seen within the last [maxAge] frames). */
    val activeTracks: Int get() = tracks.size

    /** Track ids of [boxes], in the same order. */
    fun update(boxes: List<SourceBox>): IntArray {
        val ids = IntArray(boxes.size) { -1 }
        val pairs = ArrayList<Triple<Double, Int, Int>>() // iou, track index, box index
        for ((ti, t) in tracks.withIndex()) for ((bi, b) in boxes.withIndex()) {
            val iou = t.box.iou(b)
            if (iou >= iouThreshold) pairs.add(Triple(iou, ti, bi))
        }
        pairs.sortWith(compareByDescending<Triple<Double, Int, Int>> { it.first }.thenBy { it.second }.thenBy { it.third })
        val trackUsed = BooleanArray(tracks.size)
        for ((_, ti, bi) in pairs) {
            if (trackUsed[ti] || ids[bi] != -1) continue
            trackUsed[ti] = true
            ids[bi] = tracks[ti].id
            tracks[ti].box = boxes[bi]
            tracks[ti].missed = 0
        }
        for ((ti, t) in tracks.withIndex()) if (!trackUsed[ti]) t.missed++
        tracks.removeAll { it.missed > maxAge }
        for ((bi, b) in boxes.withIndex()) {
            if (ids[bi] == -1) {
                ids[bi] = nextId
                tracks.add(Track(nextId++, b))
            }
        }
        return ids
    }

    /** Forgets every track; ids start again at 1. */
    fun reset() {
        tracks.clear()
        nextId = 1
    }
}
