package com.edgemango.app.core

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class MangoTrackerTest {
    private fun box(x: Double, y: Double, s: Double = 100.0) = SourceBox(x, y, x + s, y + s)

    private fun scored(vararg c: Pair<SourceBox, Float>) = c.toList()

    @Test
    fun hysteresisKeepsALowScoreMatchAndRejectsALowScoreNewBox() {
        val t = MangoTracker()
        val first = t.update(scored(box(0.0, 0.0) to 0.9f), 0.35f, 0.15f, 5)
        assertEquals(1, first.single().id)
        // Same mango at 0.20 (< start, >= keep) is kept; a new box far away at 0.20 is rejected.
        val out = t.update(scored(box(4.0, 2.0) to 0.20f, box(400.0, 0.0) to 0.20f), 0.35f, 0.15f, 5)
        assertEquals(listOf(1), out.map { it.id })
        assertEquals(false, out.single().held)
        assertEquals(1, t.activeTracks)
    }

    @Test
    fun aLowScoreBoxBelowKeepNeverMatches() {
        val t = MangoTracker()
        t.update(scored(box(0.0, 0.0) to 0.9f), 0.35f, 0.15f, 5)
        val out = t.update(scored(box(2.0, 2.0) to 0.10f), 0.35f, 0.15f, 5)
        assertEquals(listOf(true), out.map { it.held }) // dropped as a candidate, so the track is held instead
    }

    @Test
    fun aHighScoreNewBoxStartsATrack() {
        val t = MangoTracker()
        val out = t.update(scored(box(0.0, 0.0) to 0.40f, box(300.0, 0.0) to 0.36f), 0.35f, 0.15f, 5)
        assertEquals(listOf(1, 2), out.map { it.id })
        assertEquals(2, t.activeTracks)
    }

    @Test
    fun aMissedTrackIsHeldWithItsLastBoxForHoldFramesThenDropped() {
        val t = MangoTracker()
        val last = box(10.0, 20.0)
        t.update(scored(box(0.0, 0.0) to 0.8f), 0.35f, 0.15f, 3)
        t.update(scored(last to 0.6f), 0.35f, 0.15f, 3)
        for (missed in 1..3) {
            val held = t.update(emptyList(), 0.35f, 0.15f, 3).single()
            assertEquals(true, held.held)
            assertEquals(1, held.id)
            assertEquals(last, held.box)
            assertEquals(0.6f, held.score, 0f)
        }
        assertEquals(emptyList<TrackedBox>(), t.update(emptyList(), 0.35f, 0.15f, 3)) // missed holdFrames + 1
        assertEquals(1, t.activeTracks) // still alive (maxAge 15) so a later match keeps the id
        assertEquals(1, t.update(scored(box(12.0, 22.0) to 0.5f), 0.35f, 0.15f, 3).single().id)
    }

    @Test
    fun holdFramesIsClampedToMaxAge() {
        val t = MangoTracker(maxAge = 2)
        t.update(scored(box(0.0, 0.0) to 0.8f), 0.35f, 0.15f, 10)
        assertEquals(1, t.update(emptyList(), 0.35f, 0.15f, 10).size)
        assertEquals(1, t.update(emptyList(), 0.35f, 0.15f, 10).size)
        assertEquals(0, t.update(emptyList(), 0.35f, 0.15f, 10).size)
        assertEquals(0, t.activeTracks)
    }

    @Test
    fun idsStartAtOneAndAreStableWhileBoxesOverlap() {
        val t = MangoTracker()
        assertArrayEquals(intArrayOf(1, 2), t.update(listOf(box(0.0, 0.0), box(300.0, 0.0))))
        // Both moved a little (IoU well above 0.3) and the list order changed: ids follow the boxes.
        assertArrayEquals(intArrayOf(2, 1), t.update(listOf(box(305.0, 4.0), box(6.0, 3.0))))
    }

    @Test
    fun aBoxBelowTheIouThresholdStartsANewTrack() {
        val t = MangoTracker(iouThreshold = 0.3)
        t.update(listOf(box(0.0, 0.0)))
        // Shifted by 80 px of 100: IoU = 20/180 = 0.11.
        assertArrayEquals(intArrayOf(2), t.update(listOf(box(80.0, 0.0))))
    }

    @Test
    fun theBestOverlapWinsWhenTwoBoxesCompeteForOneTrack() {
        val t = MangoTracker()
        t.update(listOf(box(0.0, 0.0)))
        val ids = t.update(listOf(box(40.0, 0.0), box(5.0, 0.0)))
        assertEquals(1, ids[1])
        assertEquals(2, ids[0])
    }

    @Test
    fun aTrackSurvivesAGapOfMaxAgeFramesAndThenIsDropped() {
        val t = MangoTracker(maxAge = 15)
        t.update(listOf(box(0.0, 0.0)))
        repeat(15) { t.update(emptyList()) }
        assertEquals(1, t.activeTracks)
        assertArrayEquals(intArrayOf(1), t.update(listOf(box(2.0, 2.0)))) // back after 15 empty frames: same id

        repeat(16) { t.update(emptyList()) }
        assertEquals(0, t.activeTracks)
        assertArrayEquals(intArrayOf(2), t.update(listOf(box(2.0, 2.0)))) // dropped: new id, never reused
    }

    @Test
    fun aTrackFollowsAMovingMango() {
        val t = MangoTracker()
        var id = 0
        for (i in 0 until 20) {
            val got = t.update(listOf(box(i * 20.0, 0.0)))[0] // 20 px per frame: IoU 0.67 frame to frame
            if (i == 0) id = got else assertEquals(id, got)
        }
    }

    @Test
    fun resetForgetsEverything() {
        val t = MangoTracker()
        t.update(listOf(box(0.0, 0.0), box(300.0, 0.0)))
        t.reset()
        assertEquals(0, t.activeTracks)
        assertArrayEquals(intArrayOf(1), t.update(listOf(box(0.0, 0.0))))
    }
}
