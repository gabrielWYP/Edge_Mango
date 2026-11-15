package com.edgemango.app.inference

/** What the user can ask for. The session reports which one it really got (see [LiteRtSession.label]). */
enum class AcceleratorChoice(val label: String, val cpuThreads: Int = 0) {
    CPU_1("CPU (XNNPACK) x1", 1),
    CPU_2("CPU (XNNPACK) x2", 2),
    CPU_4("CPU (XNNPACK) x4", 4),
    GPU("GPU"),
    NPU("NPU (MediaTek, experimental)");

    val isCpu: Boolean get() = cpuThreads > 0

    /** Order in which creation is attempted: the request first, then progressively safer accelerators. */
    fun fallbackChain(): List<AcceleratorChoice> = when (this) {
        NPU -> listOf(NPU, GPU, CPU_4)
        GPU -> listOf(GPU, CPU_4)
        else -> listOf(this)
    }
}
