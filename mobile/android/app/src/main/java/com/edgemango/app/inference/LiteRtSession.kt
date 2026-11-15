package com.edgemango.app.inference

import android.content.Context
import com.edgemango.app.core.Inferencer
import com.edgemango.app.core.ModelSidecar
import com.edgemango.app.core.ModelTask
import com.edgemango.app.core.OutputRoles
import com.edgemango.app.core.RawOutputs
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer

/**
 * One loaded model on one accelerator (LiteRT `CompiledModel` API). Buffers are preallocated; [infer] copies the
 * input in, runs, and copies every output out as float arrays (that copy is part of the measured latency).
 *
 * Use [open] rather than the constructor: it tries the requested accelerator, verifies it with a first run and
 * falls back down [AcceleratorChoice.fallbackChain], recording why in [note].
 */
class LiteRtSession private constructor(
    private val model: CompiledModel,
    private val environment: Environment?,
    val sidecar: ModelSidecar,
    /** The accelerator that is actually running. */
    val active: AcceleratorChoice,
    val note: String,
    val loadMs: Double,
) : Inferencer, AutoCloseable {

    private val inputs: List<TensorBuffer> = model.createInputBuffers()
    private val outputs: List<TensorBuffer> = model.createOutputBuffers()
    private var roles: OutputRoles? = null

    /** Duration in ms of the three parts of the last [infer]: copy in, run, copy out. */
    @Volatile var lastWriteMs = 0.0; private set
    @Volatile var lastRunMs = 0.0; private set
    @Volatile var lastReadMs = 0.0; private set

    val label: String get() = active.label

    @Synchronized
    override fun infer(input: FloatArray): RawOutputs {
        val t0 = System.nanoTime()
        inputs[0].writeFloat(input)
        val t1 = System.nanoTime()
        model.run(inputs, outputs)
        val t2 = System.nanoTime()
        val arrays = outputs.map { it.readFloat() }
        val t3 = System.nanoTime()
        lastWriteMs = (t1 - t0) / 1e6
        lastRunMs = (t2 - t1) / 1e6
        lastReadMs = (t3 - t2) / 1e6

        val r = roles ?: resolveRoles(arrays).also { roles = it }
        return RawOutputs(arrays[r.boxes], arrays[r.cls], r.coefs?.let { arrays[it] }, r.protos?.let { arrays[it] })
    }

    private fun resolveRoles(arrays: List<FloatArray>): OutputRoles {
        val s = sidecar
        val anchors = s.decode.strides.sumOf { (s.inputSize / it) * (s.inputSize / it) }
        val protoCells = (s.inputSize / 4) * (s.inputSize / 4)
        // A model without a sidecar may be a segment model whatever the file name suggested.
        val task = if (arrays.size >= 4) ModelTask.SEGMENT else ModelTask.DETECT
        return OutputRoles.resolve(
            IntArray(arrays.size) { arrays[it].size }, task, anchors, s.numClasses, s.numPrototypes, protoCells,
            sidecarOrder = s.outputOrder.takeIf { it.isNotEmpty() },
        )
    }

    override fun close() {
        runCatching { inputs.forEach { it.close() } }
        runCatching { outputs.forEach { it.close() } }
        runCatching { model.close() }
        runCatching { environment?.close() }
    }

    companion object {
        /**
         * @param modelPath absolute path of the `.tflite`
         * @throws IllegalStateException when no accelerator in the chain could run the model
         */
        fun open(context: Context, modelPath: String, sidecar: ModelSidecar, choice: AcceleratorChoice): LiteRtSession {
            val notes = mutableListOf<String>()
            for (candidate in choice.fallbackChain()) {
                val t0 = System.nanoTime()
                var env: Environment? = null
                var model: CompiledModel? = null
                try {
                    val (options, e) = optionsFor(context, candidate)
                    env = e
                    model = if (e != null) CompiledModel.create(modelPath, options, e) else CompiledModel.create(modelPath, options)
                    val loadMs = (System.nanoTime() - t0) / 1e6
                    val note = if (candidate == choice) "" else notes.joinToString("; ") + "; running on ${candidate.label}"
                    val session = LiteRtSession(model, env, sidecar, candidate, note, loadMs)
                    // A model can load on an accelerator and still fail to run: prove it with one run.
                    session.infer(FloatArray(sidecar.inputSize * sidecar.inputSize * 3))
                    return session
                } catch (t: Throwable) {
                    notes += "${candidate.label} failed: ${t.message ?: t.javaClass.simpleName}"
                    runCatching { model?.close() }
                    runCatching { env?.close() }
                }
            }
            throw IllegalStateException(notes.joinToString("; "))
        }

        private fun optionsFor(context: Context, choice: AcceleratorChoice): Pair<CompiledModel.Options, Environment?> = when {
            choice.isCpu -> {
                val o = CompiledModel.Options(Accelerator.CPU)
                o.cpuOptions = CompiledModel.CpuOptions(numThreads = choice.cpuThreads, xnnPackFlags = null, xnnPackWeightCachePath = null)
                o to null
            }
            choice == AcceleratorChoice.GPU -> CompiledModel.Options(Accelerator.GPU) to null
            else -> {
                val provider = BuiltinNpuAcceleratorProvider(context)
                check(provider.isDeviceSupported()) { "this SoC is not supported by the bundled NPU runtime" }
                CompiledModel.Options(Accelerator.NPU, Accelerator.CPU) to Environment.create(context, provider)
            }
        }

        /** Accelerators the runtime reports as available on this device (for the settings screen). */
        fun available(context: Context): String = runCatching {
            Environment.create(context).use { env -> env.getAvailableAccelerators().joinToString { it.name } }
        }.getOrElse { "unknown (${it.message})" }
    }
}
