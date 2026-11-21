package com.edgemango.app.core

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class MangoTrackerTest {
    private fun box(x: Double, y: Double, s: Double = 100.0) = SourceBox(x, y, x + s, y + s)

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
