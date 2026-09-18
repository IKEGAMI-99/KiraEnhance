package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MnnPisaGraphContractValidatorTest {
    @Test
    fun `accepts exporter endpoint contract`() {
        val result = MnnPisaGraphContractValidator.validate(requiredTensors())

        assertTrue(result.isCompatible)
        assertTrue(result.missingEndpoints.isEmpty())
        assertTrue(result.duplicateEndpoints.isEmpty())
        assertTrue(result.unexpectedEndpoints.isEmpty())
        assertEquals(null, result.conciseProblem())
    }

    @Test
    fun `reports missing required endpoint`() {
        val tensors = requiredTensors().filterNot {
            it.graph == MnnPisaGraph.UNET &&
                it.role == MnnTensorRole.INPUT &&
                it.name == "timestep"
        }

        val result = MnnPisaGraphContractValidator.validate(tensors)

        assertFalse(result.isCompatible)
        assertEquals(
            listOf("unet/input/timestep"),
            result.missingEndpoints,
        )
        assertTrue(result.conciseProblem()?.contains("missing=unet/input/timestep") == true)
    }

    @Test
    fun `reports duplicate required endpoint`() {
        val duplicate = tensor(
            MnnPisaGraph.VAE_DECODER,
            MnnTensorRole.OUTPUT,
            "image",
        )

        val result = MnnPisaGraphContractValidator.validate(
            requiredTensors() + duplicate,
        )

        assertFalse(result.isCompatible)
        assertEquals(
            listOf("vae_decoder/output/image"),
            result.duplicateEndpoints,
        )
    }

    @Test
    fun `allows but reports extra endpoints for diagnostics`() {
        val result = MnnPisaGraphContractValidator.validate(
            requiredTensors() + tensor(
                MnnPisaGraph.UNET,
                MnnTensorRole.OUTPUT,
                "debug_output",
            ),
        )

        assertTrue(result.isCompatible)
        assertEquals(
            listOf("unet/output/debug_output"),
            result.unexpectedEndpoints,
        )
    }

    private fun requiredTensors(): List<MnnPisaTensorInfo> = listOf(
        tensor(MnnPisaGraph.VAE_ENCODER, MnnTensorRole.INPUT, "image"),
        tensor(MnnPisaGraph.VAE_ENCODER, MnnTensorRole.OUTPUT, "moments"),
        tensor(MnnPisaGraph.UNET, MnnTensorRole.INPUT, "latent"),
        tensor(MnnPisaGraph.UNET, MnnTensorRole.INPUT, "timestep"),
        tensor(MnnPisaGraph.UNET, MnnTensorRole.INPUT, "encoder_hidden_states"),
        tensor(MnnPisaGraph.UNET, MnnTensorRole.OUTPUT, "model_pred"),
        tensor(MnnPisaGraph.VAE_DECODER, MnnTensorRole.INPUT, "latent"),
        tensor(MnnPisaGraph.VAE_DECODER, MnnTensorRole.OUTPUT, "image"),
    )

    private fun tensor(
        graph: MnnPisaGraph,
        role: MnnTensorRole,
        name: String,
    ): MnnPisaTensorInfo = MnnPisaTensorInfo(
        graph = graph,
        role = role,
        name = name,
        shape = listOf(1),
        typeCode = 2,
        typeBits = 32,
        typeLanes = 1,
        dimensionType = 1,
    )
}
