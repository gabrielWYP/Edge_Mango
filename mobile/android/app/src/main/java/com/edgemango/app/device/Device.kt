package com.edgemango.app.device

import android.os.Build
import android.os.Debug
import com.edgemango.app.core.ProcStatus
import java.io.File

object DeviceInfo {
    val device: String get() = "${Build.MANUFACTURER} ${Build.MODEL}".trim()

    val soc: String
        get() = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}".trim() else Build.HARDWARE

    val android: String get() = "${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})"
}

/** One reading of the process memory counters, in MB. */
data class MemorySnapshot(
    /** `VmHWM`: peak resident set size of the whole process since start (or since the last successful [MemoryProbe.resetPeak]). */
    val peakRssMb: Double,
    /** `VmRSS`: resident set size now. */
    val rssMb: Double,
    /** `Debug.getNativeHeapAllocatedSize()`: native (malloc) heap in use, where LiteRT keeps weights and activations. */
    val nativeHeapMb: Double,
    /** Java heap in use. */
    val javaHeapMb: Double,
)

object MemoryProbe {
    private const val MB = 1024.0 * 1024.0

    /**
     * Resets the kernel's `VmHWM` high-water mark to the current RSS (`echo 5 > /proc/self/clear_refs`), so the next
     * reading covers only what happens after this call. Returns false when the kernel or SELinux refuses, in which case
     * `VmHWM` is the peak since the process started.
     */
    fun resetPeak(): Boolean = runCatching { File("/proc/self/clear_refs").writeText("5"); true }.getOrDefault(false)

    fun snapshot(): MemorySnapshot {
        val status = runCatching { File("/proc/self/status").readText() }.getOrDefault("")
        val rt = Runtime.getRuntime()
        return MemorySnapshot(
            peakRssMb = (ProcStatus.kb(status, "VmHWM") ?: 0L) / 1024.0,
            rssMb = (ProcStatus.kb(status, "VmRSS") ?: 0L) / 1024.0,
            nativeHeapMb = Debug.getNativeHeapAllocatedSize() / MB,
            javaHeapMb = (rt.totalMemory() - rt.freeMemory()) / MB,
        )
    }
}
