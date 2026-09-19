package com.ikegami99.kiraenhance.inference.mnn

data class MnnPisaGraphContractValidation(
    val missingEndpoints: List<String>,
    val duplicateEndpoints: List<String>,
    val unexpectedEndpoints: List<String>,
) {
    val isCompatible: Boolean
        get() = missingEndpoints.isEmpty() && duplicateEndpoints.isEmpty()

    fun conciseProblem(): String? {
        if (isCompatible) return null
        return buildList {
            if (missingEndpoints.isNotEmpty()) {
                add("missing=" + missingEndpoints.joinToString(","))
            }
            if (duplicateEndpoints.isNotEmpty()) {
                add("duplicate=" + duplicateEndpoints.joinToString(","))
            }
        }.joinToString(" ")
    }
}

object MnnPisaGraphContractValidator {
    fun validate(tensors: List<MnnPisaTensorInfo>): MnnPisaGraphContractValidation {
        val indexed = tensors.groupBy(::endpointKey)

        val missing = REQUIRED_ENDPOINTS
            .filterNot(indexed::containsKey)
            .map(::displayKey)

        val duplicates = REQUIRED_ENDPOINTS
            .filter { key -> indexed[key].orEmpty().size > 1 }
            .map(::displayKey)

        val unexpected = indexed.keys
            .filterNot(REQUIRED_ENDPOINTS::contains)
            .map(::displayKey)
            .sorted()

        return MnnPisaGraphContractValidation(
            missingEndpoints = missing,
            duplicateEndpoints = duplicates,
            unexpectedEndpoints = unexpected,
        )
    }

    private fun endpointKey(tensor: MnnPisaTensorInfo): EndpointKey = EndpointKey(
        graph = tensor.graph,
        role = tensor.role,
        name = tensor.name,
    )

    private fun displayKey(key: EndpointKey): String =
        key.graph.name.lowercase() + "/" +
            key.role.name.lowercase() + "/" +
            key.name

    private data class EndpointKey(
        val graph: MnnPisaGraph,
        val role: MnnTensorRole,
        val name: String,
    )

    private val REQUIRED_ENDPOINTS = listOf(
        EndpointKey(MnnPisaGraph.VAE_ENCODER, MnnTensorRole.INPUT, "image"),
        EndpointKey(MnnPisaGraph.VAE_ENCODER, MnnTensorRole.OUTPUT, "moments"),
        EndpointKey(MnnPisaGraph.UNET, MnnTensorRole.INPUT, "latent"),
        EndpointKey(MnnPisaGraph.UNET, MnnTensorRole.INPUT, "timestep"),
        EndpointKey(MnnPisaGraph.UNET, MnnTensorRole.INPUT, "encoder_hidden_states"),
        EndpointKey(MnnPisaGraph.UNET, MnnTensorRole.OUTPUT, "model_pred"),
        EndpointKey(MnnPisaGraph.VAE_DECODER, MnnTensorRole.INPUT, "latent"),
        EndpointKey(MnnPisaGraph.VAE_DECODER, MnnTensorRole.OUTPUT, "image"),
    )
}
