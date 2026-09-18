package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MnnPisaGraphShapeValidatorTest {
    @Test
    fun `accepts fixed exporter shapes`() {
        val result = MnnPisaGraphShapeValidator.validate(fixedContract())

        assertTrue(result.isCompatible)
        assertTrue(result.issues.isEmpty())
    }

    @Test
    fun `accepts dynamic spatial dimensions`() {
        val tensors = fixedContract().map { tensor ->
            if (tensor.shape.size == 4) {
                tensor.copy(shape = tensor.shape.toMutableList().apply {
                    this[2] = -1
                    this[3] = -1
                })
            } else {
                tensor
            }
        }

        assertTrue(MnnPisaGraphShapeValidator.validate(tensors).isCompatible)
    }

    @Test
    fun `rejects wrong channel contract`() {
        val tensors = fixedContract().map { tensor ->
            if (
                tensor.graph == MnnPisaGraph.VAE_ENCODER &&
                tensor.role == MnnTensorRole.OUTPUT &&
                tensor.name == "moments"
            ) {
                tensor.copy(shape = listOf(1, 4, 64, 64))
            } else {
                tensor
            }
        }

        val result = MnnPisaGraphShapeValidator.validate(tensors)

        assertFalse(result.isCompatible)
        assertTrue(result.conciseProblem()?.contains("channels expected=8 actual=4") == true)
    }

    @Test
    fun `rejects wrong prompt embedding shape`() {
        val tensors = fixedContract().map { tensor ->
            if (tensor.name == "encoder_hidden_states") {
                tensor.copy(shape = listOf(1, 77, 768))
            } else {
                tensor
            }
        }

        val result = MnnPisaGraphShapeValidator.validate(tensors)

        assertFalse(result.isCompatible)
        assertTrue(result.conciseProblem()?.contains("expected=1024 actual=768") == true)
    }

    @Test
    fun `rejects inconsistent latent spatial shape`() {
        val tensors = fixedContract().map { tensor ->
            if (
                tensor.graph == MnnPisaGraph.UNET &&
                tensor.role == MnnTensorRole.INPUT &&
                tensor.name == "latent"
            ) {
                tensor.copy(shape = listOf(1, 4, 63, 64))
            } else {
                tensor
            }
        }

        val result = MnnPisaGraphShapeValidator.validate(tensors)

        assertFalse(result.isCompatible)
        assertTrue(result.conciseProblem()?.contains("spatial dim2") == true)
    }

    private fun fixedContract(): List<MnnPisaTensorInfo> = listOf(
        tensor(MnnPisaGraph.VAE_ENCODER, MnnTensorRole.INPUT, "image", listOf(1, 3, 512, 512)),
        tensor(MnnPisaGraph.VAE_ENCODER, MnnTensorRole.OUTPUT, "moments", listOf(1, 8, 64, 64)),
        tensor(MnnPisaGraph.UNET, MnnTensorRole.INPUT, "latent", listOf(1, 4, 64, 64)),
        tensor(MnnPisaGraph.UNET, MnnTensorRole.INPUT, "timestep", listOf(1)),
        tensor(
            MnnPisaGraph.UNET,
            MnnTensorRole.INPUT,
            "encoder_hidden_states",
            listOf(1, 77, 1024),
        ),
        tensor(MnnPisaGraph.UNET, MnnTensorRole.OUTPUT, "model_pred", listOf(1, 4, 64, 64)),
        tensor(MnnPisaGraph.VAE_DECODER, MnnTensorRole.INPUT, "latent", listOf(1, 4, 64, 64)),
        tensor(MnnPisaGraph.VAE_DECODER, MnnTensorRole.OUTPUT, "image", listOf(1, 3, 512, 512)),
    )

    private fun tensor(
        graph: MnnPisaGraph,
        role: MnnTensorRole,
        name: String,
        shape: List<Int>,
    ): MnnPisaTensorInfo = MnnPisaTensorInfo(
        graph = graph,
        role = role,
        name = name,
        shape = shape,
        typeCode = 2,
        typeBits = 16,
        typeLanes = 1,
        dimensionType = 1,
    )
}
