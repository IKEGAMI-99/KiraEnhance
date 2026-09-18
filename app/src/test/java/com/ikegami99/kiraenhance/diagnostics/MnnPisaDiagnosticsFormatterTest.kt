package com.ikegami99.kiraenhance.diagnostics

import com.ikegami99.kiraenhance.inference.mnn.MnnPisaBackend
import com.ikegami99.kiraenhance.inference.mnn.MnnPisaDiagnosticsSnapshot
import com.ikegami99.kiraenhance.inference.mnn.MnnPisaGraph
import com.ikegami99.kiraenhance.inference.mnn.MnnPisaSessionInfo
import com.ikegami99.kiraenhance.inference.mnn.MnnPisaTensorInfo
import com.ikegami99.kiraenhance.inference.mnn.MnnTensorRole
import org.junit.Assert.assertEquals
import org.junit.Test

class MnnPisaDiagnosticsFormatterTest {
    @Test
    fun `formats session summary and tensor lines deterministically`() {
        val snapshot = MnnPisaDiagnosticsSnapshot(
            modelVersion = "converted-test",
            sessionInfo = MnnPisaSessionInfo(
                backend = MnnPisaBackend.OPENCL,
                gpuEnabled = true,
            ),
            tensorInfo = listOf(
                MnnPisaTensorInfo(
                    graph = MnnPisaGraph.VAE_ENCODER,
                    role = MnnTensorRole.INPUT,
                    name = "image",
                    shape = listOf(1, 3, 512, 512),
                    typeCode = 2,
                    typeBits = 16,
                    typeLanes = 1,
                    dimensionType = 1,
                ),
                MnnPisaTensorInfo(
                    graph = MnnPisaGraph.UNET,
                    role = MnnTensorRole.INPUT,
                    name = "timestep",
                    shape = emptyList(),
                    typeCode = 0,
                    typeBits = 64,
                    typeLanes = 1,
                    dimensionType = 0,
                ),
            ),
        )

        assertEquals(
            listOf(
                "modelVersion=converted-test backend=opencl gpu=true tensorCount=2",
                "graph=vae_encoder role=input name=image shape=1x3x512x512 type=2/16/1 dim=1",
                "graph=unet role=input name=timestep shape=scalar type=0/64/1 dim=0",
            ),
            MnnPisaDiagnosticsFormatter.format(snapshot),
        )
    }

    @Test
    fun `formats unavailable native diagnostics without guessing`() {
        val snapshot = MnnPisaDiagnosticsSnapshot(
            modelVersion = "local-check",
            sessionInfo = null,
            tensorInfo = null,
        )

        assertEquals(
            listOf(
                "modelVersion=local-check backend=unknown gpu=unknown tensorCount=unavailable",
            ),
            MnnPisaDiagnosticsFormatter.format(snapshot),
        )
    }
}
