package com.edgemango.app.core

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Binary fruit mask (1 = fruit) of `width x height` pixels. */
class MangoMask(val width: Int, val height: Int, val data: ByteArray, val area: Int) {
    /** Bounding box of the mask, `x2`/`y2` exclusive, or null when it is empty. */
    fun bbox(): IntBox? {
        var x1 = width
        var y1 = height
        var x2 = -1
        var y2 = -1
        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                if (data[row + x].toInt() != 0) {
                    if (x < x1) x1 = x
                    if (x > x2) x2 = x
                    if (y < y1) y1 = y
                    if (y > y2) y2 = y
                }
            }
        }
        return if (x2 < 0) null else IntBox(x1, y1, x2 + 1, y2 + 1)
    }
}

/**
 * Pure Kotlin port of `src.annotation.mango_registration.rgb_mango_mask`: yellow skin has little blue next to
 * the cyan/white sheet.
 *
 * 1. Gaussian blur, sigma = max(1, 0.005 * width), on the float32 channels.
 * 2. `ratio8 = clip(B / (G + R + 1) * 256, 0, 255)` as uint8; `lit = (G + R) > 40`.
 * 3. Otsu threshold over the `lit` pixels; `mask = (ratio8 < level) & lit`.
 * 4. Morphological OPEN with an ellipse of ~0.022 * width, then CLOSE with ~0.037 * width (odd sizes).
 * 5. Largest 8-connected component.
 *
 * Run it on a frame downscaled to about [WORK_WIDTH] px so it is cheap enough for every camera frame; the kernel
 * sizes scale with the width exactly as in Python.
 */
object MangoColorSegmenter {
    const val WORK_WIDTH = 320

    fun odd(value: Double, minimum: Int = 3): Int {
        val n = max(minimum, Math.rint(value).toInt())
        return if (n % 2 == 1) n else n + 1
    }

    fun segment(image: RgbImage): MangoMask {
        val w = image.width
        val h = image.height
        val n = w * h
        val sigma = max(1.0, 0.005 * w)
        val kernel = gaussianKernel(sigma)
        val red = FloatArray(n)
        val green = FloatArray(n)
        val blue = FloatArray(n)
        for (i in 0 until n) {
            val p = image.pixels[i]
            red[i] = ((p shr 16) and 0xFF).toFloat()
            green[i] = ((p shr 8) and 0xFF).toFloat()
            blue[i] = (p and 0xFF).toFloat()
        }
        val tmp = FloatArray(n)
        blur(red, tmp, w, h, kernel)
        blur(green, tmp, w, h, kernel)
        blur(blue, tmp, w, h, kernel)

        val ratio8 = IntArray(n)
        val lit = BooleanArray(n)
        val hist = IntArray(256)
        var litCount = 0
        for (i in 0 until n) {
            val gr = green[i] + red[i]
            val r = blue[i] / (gr + 1f) * 256f
            ratio8[i] = r.coerceIn(0f, 255f).toInt()
            if (gr > 40f) {
                lit[i] = true
                hist[ratio8[i]]++
                litCount++
            }
        }
        if (litCount == 0) return MangoMask(w, h, ByteArray(n), 0)

        val level = otsuLevel(hist, litCount)
        var mask = ByteArray(n)
        for (i in 0 until n) if (lit[i] && ratio8[i] < level) mask[i] = 1

        mask = open(mask, w, h, odd(0.022 * w))
        mask = close(mask, w, h, odd(0.037 * w))
        return largestComponent(mask, w, h)
    }

    // ---- Gaussian blur (cv2.GaussianBlur, CV_32F, BORDER_REFLECT_101) -------------------------------------

    internal fun gaussianKernel(sigma: Double): FloatArray {
        val size = Math.rint(sigma * 4 * 2 + 1).toInt() or 1
        val k = DoubleArray(size)
        val scale2 = -0.5 / (sigma * sigma)
        var sum = 0.0
        for (i in 0 until size) {
            val x = i - (size - 1) * 0.5
            k[i] = exp(scale2 * x * x)
            sum += k[i]
        }
        return FloatArray(size) { (k[it] / sum).toFloat() }
    }

    private fun reflect(i: Int, n: Int): Int {
        if (n == 1) return 0
        var v = i
        while (v < 0 || v >= n) {
            if (v < 0) v = -v
            if (v >= n) v = 2 * (n - 1) - v
        }
        return v
    }

    private fun blur(channel: FloatArray, tmp: FloatArray, w: Int, h: Int, kernel: FloatArray) {
        val r = kernel.size / 2
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var acc = 0f
                for (j in kernel.indices) acc += kernel[j] * channel[row + reflect(x + j - r, w)]
                tmp[row + x] = acc
            }
        }
        for (y in 0 until h) {
            for (x in 0 until w) {
                var acc = 0f
                for (j in kernel.indices) acc += kernel[j] * tmp[reflect(y + j - r, h) * w + x]
                channel[y * w + x] = acc
            }
        }
    }

    // ---- Otsu (OpenCV getThreshVal_Otsu_8u) ---------------------------------------------------------------

    internal fun otsuLevel(hist: IntArray, total: Int): Int {
        val scale = 1.0 / total
        var mu = 0.0
        for (i in 0 until 256) mu += i * hist[i].toDouble()
        mu *= scale
        var mu1 = 0.0
        var q1 = 0.0
        var maxSigma = 0.0
        var maxVal = 0
        val eps = 1.1920929e-7 // FLT_EPSILON
        for (i in 0 until 256) {
            val p = hist[i] * scale
            mu1 *= q1
            q1 += p
            val q2 = 1.0 - q1
            if (min(q1, q2) < eps || max(q1, q2) > 1.0 - eps) continue
            mu1 = (mu1 + i * p) / q1
            val mu2 = (mu - q1 * mu1) / q2
            val sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2)
            if (sigma > maxSigma) {
                maxSigma = sigma
                maxVal = i
            }
        }
        return maxVal
    }

    // ---- Morphology with cv2's ellipse structuring element --------------------------------------------------

    /** Per kernel row the inclusive column range `[first, last]` of `getStructuringElement(MORPH_ELLIPSE, (k, k))`. */
    internal fun ellipseRuns(k: Int): List<Pair<Int, Int>> {
        val r = k / 2
        val c = k / 2
        val invR2 = if (r != 0) 1.0 / (r.toDouble() * r) else 0.0
        return (0 until k).map { i ->
            val dy = i - r
            if (Math.abs(dy) <= r) {
                val dx = Math.rint(c * Math.sqrt((r * r - dy * dy) * invR2)).toInt()
                max(c - dx, 0) to (min(c + dx + 1, k) - 1)
            } else {
                0 to -1
            }
        }
    }

    private fun open(mask: ByteArray, w: Int, h: Int, k: Int) = morph(morph(mask, w, h, k, erode = true), w, h, k, erode = false)
    private fun close(mask: ByteArray, w: Int, h: Int, k: Int) = morph(morph(mask, w, h, k, erode = false), w, h, k, erode = true)

    /**
     * Erosion (all in-frame kernel pixels set) or dilation (any set), out-of-frame pixels ignored like OpenCV's
     * +/-inf border. Each kernel row is a horizontal run, so a row prefix sum answers it in O(1).
     */
    private fun morph(mask: ByteArray, w: Int, h: Int, k: Int, erode: Boolean): ByteArray {
        val runs = ellipseRuns(k)
        val r = k / 2
        val prefix = IntArray((w + 1) * h)
        for (y in 0 until h) {
            val base = y * (w + 1)
            for (x in 0 until w) prefix[base + x + 1] = prefix[base + x] + mask[y * w + x]
        }
        val out = ByteArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var result = erode
                for (i in runs.indices) {
                    val yy = y + i - r
                    if (yy < 0 || yy >= h) continue
                    val (first, last) = runs[i]
                    val lo = max(x + first - r, 0)
                    val hi = min(x + last - r, w - 1)
                    if (hi < lo) continue
                    val ones = prefix[yy * (w + 1) + hi + 1] - prefix[yy * (w + 1) + lo]
                    if (erode) {
                        if (ones != hi - lo + 1) { result = false; break }
                    } else if (ones > 0) {
                        result = true
                        break
                    }
                }
                out[y * w + x] = if (result) 1 else 0
            }
        }
        return out
    }

    // ---- Largest 8-connected component ------------------------------------------------------------------------

    private fun largestComponent(mask: ByteArray, w: Int, h: Int): MangoMask {
        val labels = IntArray(w * h)
        val stack = IntArray(w * h)
        var best = 0
        var bestArea = 0
        var count = 0
        for (start in mask.indices) {
            if (mask[start].toInt() == 0 || labels[start] != 0) continue
            count++
            var sp = 0
            var area = 0
            stack[sp++] = start
            labels[start] = count
            while (sp > 0) {
                val p = stack[--sp]
                area++
                val px = p % w
                val py = p / w
                for (dy in -1..1) {
                    val ny = py + dy
                    if (ny < 0 || ny >= h) continue
                    for (dx in -1..1) {
                        val nx = px + dx
                        if (nx < 0 || nx >= w) continue
                        val q = ny * w + nx
                        if (mask[q].toInt() != 0 && labels[q] == 0) {
                            labels[q] = count
                            stack[sp++] = q
                        }
                    }
                }
            }
            if (area > bestArea) {
                bestArea = area
                best = count
            }
        }
        if (count == 0) return MangoMask(w, h, mask, 0)
        val out = ByteArray(w * h)
        for (i in out.indices) if (labels[i] == best) out[i] = 1
        return MangoMask(w, h, out, bestArea)
    }

    /** Box-average downscale by integer factor `f` (>= 1) to roughly [WORK_WIDTH] px wide. */
    fun downscaleFactor(width: Int): Int = max(1, (width.toDouble() / WORK_WIDTH).roundToInt())

    fun downscale(image: RgbImage, f: Int): RgbImage {
        if (f <= 1) return image
        val w = image.width / f
        val h = image.height / f
        val out = IntArray(w * h)
        val div = f * f
        for (y in 0 until h) {
            for (x in 0 until w) {
                var r = 0
                var g = 0
                var b = 0
                for (dy in 0 until f) {
                    val row = (y * f + dy) * image.width + x * f
                    for (dx in 0 until f) {
                        val p = image.pixels[row + dx]
                        r += (p shr 16) and 0xFF
                        g += (p shr 8) and 0xFF
                        b += p and 0xFF
                    }
                }
                out[y * w + x] = (0xFF shl 24) or ((r / div) shl 16) or ((g / div) shl 8) or (b / div)
            }
        }
        return RgbImage(w, h, out)
    }
}
