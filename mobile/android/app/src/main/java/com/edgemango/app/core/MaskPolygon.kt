package com.edgemango.app.core

import kotlin.math.abs
import kotlin.math.hypot

/** Turns a binary mask into the outline of its largest blob. */
object MaskPolygon {
    /**
     * Outline of the largest 4-connected blob of [mask] (`w x h`, row-major) as a closed ring of pixel-corner coordinates,
     * flat `x0, y0, x1, y1, ...` (the ring is not repeated at the end). Coordinates are in mask-grid units: pixel `(i, j)`
     * covers `[i, i + 1] x [j, j + 1]`, so they map to frame pixels with the same affine as the mask layer. The ring is
     * simplified with Douglas-Peucker at [epsilon] grid cells. Null when the mask is empty.
     */
    fun largestContour(mask: BooleanArray, w: Int, h: Int, epsilon: Double = 0.75): DoubleArray? {
        val blob = largestBlob(mask, w, h) ?: return null
        val ring = outerRing(blob, w, h)
        if (ring.size < 3) return null
        val simplified = simplify(dropCollinear(ring), epsilon)
        if (simplified.size < 3) return null
        return DoubleArray(simplified.size * 2) { i -> simplified[i / 2].let { if (i % 2 == 0) it.first.toDouble() else it.second.toDouble() } }
    }

    /** Membership of the largest 4-connected blob, or null when the mask is empty. */
    private fun largestBlob(mask: BooleanArray, w: Int, h: Int): BooleanArray? {
        val label = IntArray(w * h)
        var best = 0
        var bestSize = 0
        var next = 0
        val stack = IntArray(w * h)
        for (start in 0 until w * h) {
            if (!mask[start] || label[start] != 0) continue
            next++
            var sp = 0
            var size = 0
            stack[sp++] = start
            label[start] = next
            while (sp > 0) {
                val p = stack[--sp]
                size++
                val x = p % w
                val y = p / w
                if (x > 0 && mask[p - 1] && label[p - 1] == 0) { label[p - 1] = next; stack[sp++] = p - 1 }
                if (x < w - 1 && mask[p + 1] && label[p + 1] == 0) { label[p + 1] = next; stack[sp++] = p + 1 }
                if (y > 0 && mask[p - w] && label[p - w] == 0) { label[p - w] = next; stack[sp++] = p - w }
                if (y < h - 1 && mask[p + w] && label[p + w] == 0) { label[p + w] = next; stack[sp++] = p + w }
            }
            if (size > bestSize) { bestSize = size; best = next }
        }
        if (best == 0) return null
        return BooleanArray(w * h) { label[it] == best }
    }

    /**
     * Boundary of the blob as directed unit edges (clockwise around the inside, image coordinates), chained into closed
     * rings; the ring with the largest area is the outer boundary (holes are smaller).
     */
    private fun outerRing(blob: BooleanArray, w: Int, h: Int): List<Pair<Int, Int>> {
        val stride = w + 1
        val next = HashMap<Int, MutableList<Int>>()
        fun edge(x1: Int, y1: Int, x2: Int, y2: Int) {
            next.getOrPut(y1 * stride + x1) { ArrayList(1) }.add(y2 * stride + x2)
        }
        fun inside(x: Int, y: Int) = x in 0 until w && y in 0 until h && blob[y * w + x]
        for (y in 0 until h) for (x in 0 until w) {
            if (!blob[y * w + x]) continue
            if (!inside(x, y - 1)) edge(x, y, x + 1, y)
            if (!inside(x + 1, y)) edge(x + 1, y, x + 1, y + 1)
            if (!inside(x, y + 1)) edge(x + 1, y + 1, x, y + 1)
            if (!inside(x - 1, y)) edge(x, y + 1, x, y)
        }
        var best: List<Pair<Int, Int>> = emptyList()
        var bestArea = -1.0
        for (startKey in next.keys.toList()) {
            while (next[startKey]?.isNotEmpty() == true) {
                val ring = ArrayList<Pair<Int, Int>>()
                var cur = startKey
                do {
                    ring.add(cur % stride to cur / stride)
                    cur = next.getValue(cur).removeAt(next.getValue(cur).size - 1)
                } while (cur != startKey)
                val area = abs(signedArea(ring))
                if (area > bestArea) { bestArea = area; best = ring }
            }
        }
        return best
    }

    private fun signedArea(r: List<Pair<Int, Int>>): Double {
        var a = 0.0
        for (i in r.indices) {
            val (x1, y1) = r[i]
            val (x2, y2) = r[(i + 1) % r.size]
            a += x1.toDouble() * y2 - x2.toDouble() * y1
        }
        return a / 2
    }

    /** Removes vertices that lie on the straight segment between their neighbours. */
    private fun dropCollinear(r: List<Pair<Int, Int>>): List<Pair<Int, Int>> {
        val out = ArrayList<Pair<Int, Int>>()
        for (i in r.indices) {
            val (px, py) = r[(i - 1 + r.size) % r.size]
            val (cx, cy) = r[i]
            val (nx, ny) = r[(i + 1) % r.size]
            val cross = (cx - px).toLong() * (ny - cy) - (cy - py).toLong() * (nx - cx)
            if (cross != 0L) out.add(r[i])
        }
        return out
    }

    /** Douglas-Peucker on a closed ring: split at the vertex farthest from vertex 0, simplify both open chains. */
    private fun simplify(ring: List<Pair<Int, Int>>, epsilon: Double): List<Pair<Int, Int>> {
        if (ring.size <= 4 || epsilon <= 0.0) return ring
        var far = 0
        var farD = -1.0
        for (i in 1 until ring.size) {
            val d = hypot((ring[i].first - ring[0].first).toDouble(), (ring[i].second - ring[0].second).toDouble())
            if (d > farD) { farD = d; far = i }
        }
        val a = chain(ring, 0, far, epsilon)
        val b = chain(ring + listOf(ring[0]), far, ring.size, epsilon)
        return a.dropLast(1) + b.dropLast(1)
    }

    /** Douglas-Peucker on `pts[from..to]` (inclusive ends, both kept). */
    private fun chain(pts: List<Pair<Int, Int>>, from: Int, to: Int, epsilon: Double): List<Pair<Int, Int>> {
        if (to - from < 2) return listOf(pts[from], pts[to])
        val (x1, y1) = pts[from]
        val (x2, y2) = pts[to]
        var worst = -1.0
        var at = from
        for (i in from + 1 until to) {
            val d = distanceToSegment(pts[i].first.toDouble(), pts[i].second.toDouble(), x1.toDouble(), y1.toDouble(), x2.toDouble(), y2.toDouble())
            if (d > worst) { worst = d; at = i }
        }
        if (worst <= epsilon) return listOf(pts[from], pts[to])
        return chain(pts, from, at, epsilon).dropLast(1) + chain(pts, at, to, epsilon)
    }

    private fun distanceToSegment(px: Double, py: Double, ax: Double, ay: Double, bx: Double, by: Double): Double {
        val dx = bx - ax
        val dy = by - ay
        val len2 = dx * dx + dy * dy
        if (len2 == 0.0) return hypot(px - ax, py - ay)
        val t = (((px - ax) * dx + (py - ay) * dy) / len2).coerceIn(0.0, 1.0)
        return hypot(px - (ax + t * dx), py - (ay + t * dy))
    }
}
