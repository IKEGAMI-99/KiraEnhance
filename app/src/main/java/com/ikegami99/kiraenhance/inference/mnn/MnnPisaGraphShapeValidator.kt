package com.ikegami99.kiraenhance.inference.mnn

data class MnnPisaGraphShapeValidation(
    val issues: List<String>,
) {
    val isCompatible: Boolean
        get() = issues.isEmpty()

    fun conciseProblem(): String? =
        issues.takeIf { it.isNotEmpty() }?.joinToString("; ")
}

object MnnPisaGraphShapeValidator {
    fun validate(tensors: List<MnnPisaTensorInfo>): MnnPisaGraphShapeValidation {
        val issues = mutableListOf<String>()

        val encoderInput = endpoint(
            tensors,
            MnnPisaGraph.VAE_ENCODER,
            MnnTensorRole.INPUT,
            "image",
        )
        val encoderOutput = endpoint(
            tensors,
            MnnPisaGraph.VAE_ENCODER,
            MnnTensorRole.OUTPUT,
            "moments",
        )
        val unetLatent = endpoint(
            tensors,
            MnnPisaGraph.UNET,
            MnnTensorRole.INPUT,
            "latent",
        )
        val timestep = endpoint(
            tensors,
            MnnPisaGraph.UNET,
            MnnTensorRole.INPUT,
            "timestep",
        )
        val prompt = endpoint(
            tensors,
            MnnPisaGraph.UNET,
            MnnTensorRole.INPUT,
            "encoder_hidden_states",
        )
        val modelPred = endpoint(
            tensors,
            MnnPisaGraph.UNET,
            MnnTensorRole.OUTPUT,
            "model_pred",
        )
        val decoderLatent = endpoint(
            tensors,
            MnnPisaGraph.VAE_DECODER,
            MnnTensorRole.INPUT,
            "latent",
        )
        val decoderImage = endpoint(
            tensors,
            MnnPisaGraph.VAE_DECODER,
            MnnTensorRole.OUTPUT,
            "image",
        )

        validateNchw(encoderInput, "vae_encoder/input/image", 3, issues)
        validateNchw(encoderOutput, "vae_encoder/output/moments", 8, issues)
        validateNchw(unetLatent, "unet/input/latent", 4, issues)
        validateShape(timestep, "unet/input/timestep", listOf(1), issues)
        validateShape(prompt, "unet/input/encoder_hidden_states", listOf(1, 77, 1024), issues)
        validateNchw(modelPred, "unet/output/model_pred", 4, issues)
        validateNchw(decoderLatent, "vae_decoder/input/latent", 4, issues)
        validateNchw(decoderImage, "vae_decoder/output/image", 3, issues)

        if (issues.isEmpty()) {
            validateSpatialRelationships(
                encoderInput = encoderInput,
                encoderOutput = encoderOutput,
                unetLatent = unetLatent,
                modelPred = modelPred,
                decoderLatent = decoderLatent,
                decoderImage = decoderImage,
                issues = issues,
            )
        }

        return MnnPisaGraphShapeValidation(issues)
    }

    private fun endpoint(
        tensors: List<MnnPisaTensorInfo>,
        graph: MnnPisaGraph,
        role: MnnTensorRole,
        name: String,
    ): MnnPisaTensorInfo? = tensors.singleOrNull {
        it.graph == graph && it.role == role && it.name == name
    }

    private fun validateNchw(
        tensor: MnnPisaTensorInfo?,
        label: String,
        channels: Int,
        issues: MutableList<String>,
    ) {
        if (tensor == null) return
        if (tensor.shape.size != 4) {
            issues += "$label rank expected=4 actual=" + tensor.shape.size
            return
        }
        validateFixedDimension(tensor.shape[0], 1, "$label batch", issues)
        validateFixedDimension(tensor.shape[1], channels, "$label channels", issues)
    }

    private fun validateShape(
        tensor: MnnPisaTensorInfo?,
        label: String,
        expected: List<Int>,
        issues: MutableList<String>,
    ) {
        if (tensor == null) return
        if (tensor.shape.size != expected.size) {
            issues += "$label rank expected=" + expected.size + " actual=" + tensor.shape.size
            return
        }
        expected.indices.forEach { index ->
            validateFixedDimension(
                actual = tensor.shape[index],
                expected = expected[index],
                label = "$label dim$index",
                issues = issues,
            )
        }
    }

    private fun validateFixedDimension(
        actual: Int,
        expected: Int,
        label: String,
        issues: MutableList<String>,
    ) {
        if (actual > 0 && actual != expected) {
            issues += "$label expected=$expected actual=$actual"
        }
    }

    private fun validateSpatialRelationships(
        encoderInput: MnnPisaTensorInfo?,
        encoderOutput: MnnPisaTensorInfo?,
        unetLatent: MnnPisaTensorInfo?,
        modelPred: MnnPisaTensorInfo?,
        decoderLatent: MnnPisaTensorInfo?,
        decoderImage: MnnPisaTensorInfo?,
        issues: MutableList<String>,
    ) {
        val imageH = encoderInput?.shape?.getOrNull(2)
        val imageW = encoderInput?.shape?.getOrNull(3)
        val latentH = encoderOutput?.shape?.getOrNull(2)
        val latentW = encoderOutput?.shape?.getOrNull(3)

        if (imageH != null && latentH != null && imageH > 0 && latentH > 0) {
            if (imageH % 8 != 0 || imageH / 8 != latentH) {
                issues += "vae spatial height expected image/8"
            }
        }
        if (imageW != null && latentW != null && imageW > 0 && latentW > 0) {
            if (imageW % 8 != 0 || imageW / 8 != latentW) {
                issues += "vae spatial width expected image/8"
            }
        }

        val latentPeers = listOf(
            "unet/input/latent" to unetLatent,
            "unet/output/model_pred" to modelPred,
            "vae_decoder/input/latent" to decoderLatent,
        )
        latentPeers.forEach { (label, tensor) ->
            validateSameSpatial(
                reference = encoderOutput,
                candidate = tensor,
                label = label,
                issues = issues,
            )
        }

        validateSameSpatial(
            reference = encoderInput,
            candidate = decoderImage,
            label = "vae_decoder/output/image",
            issues = issues,
        )
    }

    private fun validateSameSpatial(
        reference: MnnPisaTensorInfo?,
        candidate: MnnPisaTensorInfo?,
        label: String,
        issues: MutableList<String>,
    ) {
        if (reference == null || candidate == null) return
        for (index in 2..3) {
            val expected = reference.shape.getOrNull(index) ?: continue
            val actual = candidate.shape.getOrNull(index) ?: continue
            if (expected > 0 && actual > 0 && expected != actual) {
                issues += "$label spatial dim$index expected=$expected actual=$actual"
            }
        }
    }
}
