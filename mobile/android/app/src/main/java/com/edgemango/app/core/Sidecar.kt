package com.edgemango.app.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.longOrNull

enum class ModelTask { DETECT, SEGMENT }

/** Which export contract a model follows: this project's raw-head exporter or a stock Ultralytics TFLite export. */
enum class ModelFormat { EDGE_MANGO, ULTRALYTICS }

/** Memory layout of the single Ultralytics output tensor: `(1, 4+nc, N)` or `(1, N, 4+nc)`. */
enum class OutputLayout { AUTO, CHANNELS_FIRST, CHANNELS_LAST }

data class DecodeSettings(
    val confThreshold: Float = 0.25f,
    val nmsIouThreshold: Float = 0.5f,
    val maxDetections: Int = 300,
    val strides: List<Int> = listOf(8, 16, 32),
)

/**
 * The metadata JSON that sits next to every `.tflite` (written by `scripts/export_student.py`). Parsed leniently:
 * only `task` and the input size are required, everything else falls back to the documented defaults.
 */
data class ModelSidecar(
    val task: ModelTask,
    val inputSize: Int,
    val classNames: List<String> = listOf("mango", "damage"),
    val normalization: Normalization = Normalization.IMAGENET,
    val padValue: Int = 114,
    val decode: DecodeSettings = DecodeSettings(),
    val outputOrder: List<String> = emptyList(),
    val numPrototypes: Int = 32,
    val numParameters: Long? = null,
    val flops: Double? = null,
    val precision: String? = null,
    val fileSizeBytes: Long? = null,
    val modelName: String? = null,
    val format: ModelFormat = ModelFormat.EDGE_MANGO,
    val outputLayout: OutputLayout = OutputLayout.AUTO,
) {
    val numClasses: Int get() = classNames.size

    val isUltralytics: Boolean get() = format == ModelFormat.ULTRALYTICS

    fun decodeParams() = DecodeParams(
        confThreshold = decode.confThreshold,
        nmsIouThreshold = decode.nmsIouThreshold,
        maxDetections = decode.maxDetections,
        imageSize = inputSize,
        strides = decode.strides,
        numClasses = numClasses,
    )

    companion object {
        private val json = Json { ignoreUnknownKeys = true; isLenient = true }

        /** Sidecar assumed when a model comes without one: the documented contract of the exporter. */
        fun default(task: ModelTask) = ModelSidecar(task = task, inputSize = 640)

        fun parse(text: String): ModelSidecar {
            val root = json.parseToJsonElement(text).jsonObject
            val task = when (root.string("task")?.lowercase()) {
                "detect" -> ModelTask.DETECT
                "segment" -> ModelTask.SEGMENT
                null -> throw IllegalArgumentException("sidecar has no \"task\"")
                else -> throw IllegalArgumentException("unknown sidecar task \"${root.string("task")}\"")
            }
            val format = when (root.string("format")?.lowercase()) {
                null, "edge_mango" -> ModelFormat.EDGE_MANGO
                "ultralytics" -> ModelFormat.ULTRALYTICS
                else -> throw IllegalArgumentException("unknown sidecar format \"${root.string("format")}\"")
            }
            val layout = when (root.string("output_layout")?.lowercase()) {
                null, "auto" -> OutputLayout.AUTO
                "channels_first" -> OutputLayout.CHANNELS_FIRST
                "channels_last" -> OutputLayout.CHANNELS_LAST
                else -> throw IllegalArgumentException("unknown output_layout \"${root.string("output_layout")}\"")
            }
            val inputSize = inputSizeOf(root) ?: throw IllegalArgumentException("sidecar has no input size")
            val pre = root.obj("preprocessing")
            val norm = pre?.obj("normalization")
            val mean = norm?.doubles("mean")
            val std = norm?.doubles("std")
            val dec = root.obj("decode")
            val levels = root.obj("anchors")?.get("levels") as? JsonArray
            val strides = levels?.mapNotNull { (it as? JsonObject)?.int("stride") }?.takeIf { it.isNotEmpty() } ?: listOf(8, 16, 32)
            val modelObj = root.obj("model")
            val outputs = (root["outputs"] as? JsonArray)?.mapNotNull { (it as? JsonObject)?.string("logical_name") } ?: emptyList()
            val classNames = (root["class_names"] as? JsonArray)?.mapNotNull { (it as? JsonPrimitive)?.contentOrNull }
            val defaultNames = if (format == ModelFormat.ULTRALYTICS) listOf("mango") else listOf("mango", "damage")
            // Ultralytics models take [0,1] pixels with no mean/std; this project's exporter takes ImageNet statistics.
            val defaultNorm = if (format == ModelFormat.ULTRALYTICS) Normalization.UNIT else Normalization.IMAGENET
            return ModelSidecar(
                task = task,
                inputSize = inputSize,
                classNames = classNames?.takeIf { it.isNotEmpty() } ?: defaultNames,
                normalization = if (mean != null && std != null && mean.size == 3 && std.size == 3) Normalization(mean, std) else defaultNorm,
                padValue = pre?.obj("letterbox")?.int("pad_value") ?: 114,
                decode = DecodeSettings(
                    confThreshold = dec?.double("conf_threshold")?.toFloat() ?: 0.25f,
                    nmsIouThreshold = dec?.obj("nms")?.double("iou_threshold")?.toFloat() ?: 0.5f,
                    maxDetections = dec?.int("max_detections") ?: 300,
                    strides = strides,
                ),
                outputOrder = outputs,
                numPrototypes = root.obj("segmentation")?.int("num_prototypes") ?: 32,
                numParameters = findNumber(root, modelObj, PARAM_KEYS)?.toLong(),
                flops = findFlops(root, modelObj),
                precision = modelObj?.string("precision") ?: root.string("precision"),
                fileSizeBytes = modelObj?.long("file_size_bytes") ?: root.long("file_size_bytes"),
                modelName = modelObj?.string("file") ?: modelObj?.string("name"),
                format = format,
                outputLayout = layout,
            )
        }

        private val PARAM_KEYS = listOf("num_parameters", "num_params", "params", "parameters")
        private val SCOPES = listOf("model", "complexity", "stats", "metrics", "profile")

        private fun scopes(root: JsonObject, model: JsonObject?): List<JsonObject> =
            listOfNotNull(root, model) + SCOPES.filter { it != "model" }.mapNotNull { root.obj(it) }

        private fun findNumber(root: JsonObject, model: JsonObject?, keys: List<String>): Double? {
            for (scope in scopes(root, model)) for (k in keys) scope.double(k)?.let { return it }
            for (scope in scopes(root, model)) scope.double("params_m")?.let { return it * 1e6 }
            return null
        }

        /** FLOPs from whichever spelling the exporter used; 1 MAC counts as 2 FLOPs. */
        private fun findFlops(root: JsonObject, model: JsonObject?): Double? {
            for (scope in scopes(root, model)) {
                scope.double("flops")?.let { return it }
                scope.double("gflops")?.let { return it * 1e9 }
                scope.double("flops_g")?.let { return it * 1e9 }
                scope.double("macs")?.let { return it * 2 }
                scope.double("gmacs")?.let { return it * 2e9 }
            }
            return null
        }

        private fun inputSizeOf(root: JsonObject): Int? {
            val shape = (root.obj("input")?.get("shape") as? JsonArray)?.mapNotNull { (it as? JsonPrimitive)?.intOrNull }
            if (shape != null && shape.size == 4) return if (shape[3] == 3) shape[1] else shape[2]
            root.obj("preprocessing")?.obj("letterbox")?.int("target_size")?.let { return it }
            return root.int("input_size") ?: root.int("image_size")
        }

        private fun JsonObject.obj(key: String): JsonObject? = this[key] as? JsonObject
        private fun JsonObject.string(key: String): String? = (this[key] as? JsonPrimitive)?.contentOrNull
        private fun JsonObject.int(key: String): Int? = (this[key] as? JsonPrimitive)?.intOrNull
        private fun JsonObject.long(key: String): Long? = (this[key] as? JsonPrimitive)?.longOrNull
        private fun JsonObject.double(key: String): Double? = (this[key] as? JsonPrimitive)?.doubleOrNull
        private fun JsonObject.doubles(key: String): List<Double>? =
            (this[key] as? JsonArray)?.mapNotNull { (it as? JsonPrimitive)?.doubleOrNull }
    }
}

/** Which of the runtime's output tensors is which. Resolved by element count; the sidecar order is only a tie-break. */
data class OutputRoles(val boxes: Int, val cls: Int, val coefs: Int?, val protos: Int?) {
    companion object {
        fun resolve(
            sizes: IntArray,
            task: ModelTask,
            anchors: Int,
            numClasses: Int,
            numPrototypes: Int,
            protoCells: Int,
            sidecarOrder: List<String>? = null,
        ): OutputRoles {
            val names = if (task == ModelTask.SEGMENT) listOf("boxes_raw", "cls_logits", "mask_coefs", "protos") else listOf("boxes_raw", "cls_logits")
            val expected = listOf(anchors * 4, anchors * numClasses, anchors * numPrototypes, numPrototypes * protoCells).take(names.size)
            check(sizes.size >= names.size) { "model has ${sizes.size} outputs, a ${task.name.lowercase()} model needs ${names.size}" }
            val picks = names.indices.map { role ->
                val matches = sizes.indices.filter { sizes[it] == expected[role] }
                when {
                    matches.size == 1 -> matches[0]
                    else -> {
                        val order = sidecarOrder?.indexOf(names[role])?.takeIf { it >= 0 } ?: role
                        check(order < sizes.size) { "cannot place output ${names[role]}" }
                        order
                    }
                }
            }
            check(picks.toSet().size == picks.size) {
                "output sizes ${sizes.toList()} do not identify the outputs ${names.joinToString()} (expected sizes $expected)"
            }
            return OutputRoles(picks[0], picks[1], picks.getOrNull(2), picks.getOrNull(3))
        }
    }
}
