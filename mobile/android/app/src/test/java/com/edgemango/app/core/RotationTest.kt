package com.edgemango.app.core

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test

class RotationTest {
    // 3 wide x 2 high:   1 2 3
    //                    4 5 6
    private val src = intArrayOf(1, 2, 3, 4, 5, 6)

    @Test
    fun zeroDegreesIsTheIdentityAndKeepsTheBuffer() {
        val out = Rotation.clockwise(src, 3, 2, 0, IntArray(6))
        assertSame(src, out.pixels)
        assertEquals(3, out.width)
    }

    @Test
    fun ninetyClockwise() {
        // 4 1
        // 5 2
        // 6 3
        val out = Rotation.clockwise(src, 3, 2, 90, IntArray(6))
        assertEquals(2, out.width)
        assertEquals(3, out.height)
        assertArrayEquals(intArrayOf(4, 1, 5, 2, 6, 3), out.pixels.copyOf(6))
    }

    @Test
    fun oneEighty() {
        val out = Rotation.clockwise(src, 3, 2, 180, IntArray(6))
        assertArrayEquals(intArrayOf(6, 5, 4, 3, 2, 1), out.pixels.copyOf(6))
    }

    @Test
    fun twoSeventyClockwise() {
        // 3 6
        // 2 5
        // 1 4
        val out = Rotation.clockwise(src, 3, 2, 270, IntArray(6))
        assertEquals(2, out.width)
        assertArrayEquals(intArrayOf(3, 6, 2, 5, 1, 4), out.pixels.copyOf(6))
    }
}
