package com.edgemango.app.core

/**
 * Decoder for the single output tensor of a stock Ultralytics (YOLOv8) TFLite export.
 *
 * Contract assumed (see `docs/deployment/mobile-export.md`):
 * - one tensor `(1, 4+nc, N)` (channels first, the usual export) or `(1, N, 4+nc)` (channels last), flattened row-major;
 * - per anchor `cx, cy, w, h, score_0 .. score_{nc-1}`; scores are already sigmoid probabilities;
 * - boxes are normalized to [0,1] of the model input, or already in input pixels. Told apart by magnitude: when the
 *   largest box value over all anchors is <= [NORMALIZED_MAX] the boxes are normalized and scaled by the input size
 *   (a pixel-space export has anchors with `cx` in the hundreds);
 * - no NMS inside: one candidate per anchor (best class, like Ultralytics' default), then greedy NMS ([Decoder.nms]).
 *
 * Detections come back in letterboxed-input pixels, exactly like [Decoder.decode], so [Detection.toSourceBox] applies.
 */
object UltralyticsDecoder {
    /** Box values up to this magnitude are read as normalized coordinates. */
    const val NORMALIZED_MAX = 2f

    /**
     * @param keepClasses when not empty, only these class columns count: an anchor's score is the maximum over them and
     * the detection is relabelled as class 0. Lets a stock multi-class model (e.g. COCO banana/apple/orange as "mango")
     * act as a single-class gate. Ids must lie in `0 until nc`. Empty keeps the best class over all classes.
     */
    fun decode(
        raw: FloatArray,
        params: DecodeParams,
        layout: OutputLayout = OutputLayout.AUTO,
        keepClasses: List<Int> = emptyList(),
    ): List<Detection> {
        val nc = params.numClasses
        val kept = keepClasses.distinct().sorted()
        if (kept.isNotEmpty()) {
            require(kept.all { it in 0 until nc }) { "keep_classes $keepClasses has ids outside 0 until $nc" }
        }
        val k = 4 + nc
        require(raw.size >= k && raw.size % k == 0) { "output has ${raw.size} values, not a multiple of ${k} (4 + $nc classes)" }
        val n = raw.size / k
        val channelsFirst = when (layout) {
            OutputLayout.CHANNELS_FIRST -> true
            OutputLayout.CHANNELS_LAST -> false
            OutputLayout.AUTO -> looksChannelsFirst(raw, n, nc)
        }
        fun at(anchor: Int, channel: Int): Float = if (channelsFirst) raw[channel * n + anchor] else raw[anchor * k + channel]

        var maxBox = 0f
        for (a in 0 until n) for (c in 0 until 4) {
            val v = at(a, c)
            if (v > maxBox) maxBox = v
        }
        val factor = if (maxBox <= NORMALIZED_MAX) params.imageSize.toFloat() else 1f

        val cand = ArrayList<Detection>(256)
        for (a in 0 until n) {
            var best = 0
            var bestScore: Float
            if (kept.isEmpty()) {
                bestScore = at(a, 4)
                for (c in 1 until nc) {
                    val s = at(a, 4 + c)
                    if (s > bestScore) { bestScore = s; best = c }
                }
            } else {
                bestScore = Float.NEGATIVE_INFINITY
                for (c in kept) bestScore = maxOf(bestScore, at(a, 4 + c))
            }
            if (bestScore < params.confThreshold) continue
            cand.add(Detection(best, bestScore, at(a, 0) * factor, at(a, 1) * factor, at(a, 2) * factor, at(a, 3) * factor, a))
        }
        if (cand.isEmpty()) return emptyList()
        cand.sortByDescending { it.score }
        val pool = if (params.maxCandidates > 0 && cand.size > params.maxCandidates) cand.subList(0, params.maxCandidates) else cand
        return Decoder.nms(pool, params)
    }

    /**
     * Layout guess when the sidecar does not say. Reading the data in the wrong layout puts boxes and scores in the same
     * "row", so (1) a reading whose scores leave [0,1] or whose sizes are negative is wrong, and (2) when both readings
     * are plausible (normalized boxes) the right one has the lower mean score, because most anchors are background
     * while the misread rows are mostly box coordinates. Ties fall to channels first, the layout Ultralytics exports.
     * Set `output_layout` in the sidecar to skip the guess.
     */
    private fun looksChannelsFirst(raw: FloatArray, n: Int, nc: Int): Boolean {
        val k = 4 + nc
        fun stats(channelsFirst: Boolean): Pair<Boolean, Double> {
            var valid = true
            var sum = 0.0
            for (a in 0 until n) {
                for (c in 2 until k) {
                    val v = if (channelsFirst) raw[c * n + a] else raw[a * k + c]
                    if (v < 0f || (c >= 4 && v > 1f)) valid = false
                    if (c >= 4) sum += v
                }
            }
            return valid to sum / (n.toDouble() * nc)
        }
        val (firstValid, firstMean) = stats(true)
        val (lastValid, lastMean) = stats(false)
        return when {
            firstValid != lastValid -> firstValid
            else -> firstMean <= lastMean
        }
    }
}
