package com.edgemango.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class MaskPolygonTest {
    private fun mask(w: Int, h: Int, fill: (Int, Int) -> Boolean) = BooleanArray(w * h) { fill(it % w, it / w) }

    private fun area(p: DoubleArray): Double {
        var a = 0.0
        val n = p.size / 2
        for (i in 0 until n) {
            val j = (i + 1) % n
            a += p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1]
        }
        return abs(a) / 2
    }

    @Test
    fun emptyMaskHasNoContour() {
        assertNull(MaskPolygon.largestContour(BooleanArray(100), 10, 10))
    }

    @Test
    fun rectangleCollapsesToItsFourCorners() {
        val m = mask(20, 20) { x, y -> x in 3 until 9 && y in 5 until 12 }
        val p = MaskPolygon.largestContour(m, 20, 20)!!
        assertEquals(8, p.size)
        assertEquals(6.0 * 7.0, area(p), 1e-9)
        val xs = (p.indices step 2).map { p[it] }
        val ys = (1 until p.size step 2).map { p[it] }
        assertEquals(3.0, xs.min(), 0.0); assertEquals(9.0, xs.max(), 0.0)
        assertEquals(5.0, ys.min(), 0.0); assertEquals(12.0, ys.max(), 0.0)
    }

    @Test
    fun keepsOnlyTheLargestBlob() {
        val m = mask(30, 30) { x, y -> (x in 2 until 6 && y in 2 until 6) || (x in 15 until 25 && y in 10 until 20) }
        val p = MaskPolygon.largestContour(m, 30, 30)!!
        assertEquals(100.0, area(p), 1e-9)
    }

    @Test
    fun aHoleDoesNotChangeTheOuterOutline() {
        val m = mask(20, 20) { x, y -> x in 2 until 18 && y in 2 until 18 && !(x in 8 until 12 && y in 8 until 12) }
        val p = MaskPolygon.largestContour(m, 20, 20)!!
        assertEquals(16.0 * 16.0, area(p), 1e-9)
    }

    @Test
    fun aDiscIsSimplifiedButStaysCloseToTheMask() {
        val m = mask(60, 60) { x, y -> (x - 30) * (x - 30) + (y - 30) * (y - 30) <= 20 * 20 }
        val exact = m.count { it }.toDouble()
        val p = MaskPolygon.largestContour(m, 60, 60)!!
        assertTrue("simplified to few vertices", p.size / 2 < 80)
        assertTrue("area within 4%", abs(area(p) - exact) / exact < 0.04)
        assertNotNull(p)
    }

    @Test
    fun aSinglePixelIsASquare() {
        val m = mask(5, 5) { x, y -> x == 2 && y == 3 }
        val p = MaskPolygon.largestContour(m, 5, 5)!!
        assertEquals(1.0, area(p), 1e-9)
    }
}
