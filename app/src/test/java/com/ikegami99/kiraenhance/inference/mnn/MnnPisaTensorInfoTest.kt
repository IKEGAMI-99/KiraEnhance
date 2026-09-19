package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MnnPisaTensorInfoTest {
    @Test
    fun `decodes graph tensor metadata`() {
        val raw = listOf(
            "graph=vae_encoder;role=input;nameHex=73616d706c65;shape=1,3,64,64;type=2,32,1;dim=1",
            "graph=unet;role=input;nameHex=74696d6573746570;shape=1;type=2,32,1;dim=0",
            "graph=vae_decoder;role=output;nameHex=e794bbe5838f;shape=;type=2,16,1;dim=2",
        ).joinToString("\n")

        assertEquals(
            listOf(
                MnnPisaTensorInfo(
                    graph = MnnPisaGraph.VAE_ENCODER,
                    role = MnnTensorRole.INPUT,
                    name = "sample",
                    shape = listOf(1, 3, 64, 64),
                    typeCode = 2,
                    typeBits = 32,
                    typeLanes = 1,
                    dimensionType = 1,
                ),
                MnnPisaTensorInfo(
                    graph = MnnPisaGraph.UNET,
                    role = MnnTensorRole.INPUT,
                    name = "timestep",
                    shape = listOf(1),
                    typeCode = 2,
                    typeBits = 32,
                    typeLanes = 1,
                    dimensionType = 0,
                ),
                MnnPisaTensorInfo(
                    graph = MnnPisaGraph.VAE_DECODER,
                    role = MnnTensorRole.OUTPUT,
                    name = "画像",
                    shape = emptyList(),
                    typeCode = 2,
                    typeBits = 16,
                    typeLanes = 1,
                    dimensionType = 2,
                ),
            ),
            MnnPisaTensorInfoCodec.decode(raw),
        )
    }

    @Test
    fun `blank payload is a valid empty tensor list`() {
        assertEquals(emptyList<MnnPisaTensorInfo>(), MnnPisaTensorInfoCodec.decode(""))
    }

    @Test
    fun `rejects malformed graph metadata`() {
        val invalid = listOf(
            "graph=unknown;role=input;nameHex=61;shape=1;type=2,32,1;dim=0",
            "graph=unet;role=side;nameHex=61;shape=1;type=2,32,1;dim=0",
            "graph=unet;role=input;nameHex=6;shape=1;type=2,32,1;dim=0",
            "graph=unet;role=input;nameHex=zz;shape=1;type=2,32,1;dim=0",
            "graph=unet;role=input;nameHex=61;shape=x;type=2,32,1;dim=0",
            "graph=unet;role=input;nameHex=61;shape=1;type=2,32;dim=0",
            "graph=unet;role=input;nameHex=61;shape=1;type=2,32,1;dim=x",
            "graph=unet;role=input;nameHex=61;shape=1;type=2,32,1",
            "graph=unet;role=input;nameHex=61;shape=1;type=2,32,1;dim=0;dim=1",
        )

        invalid.forEach { raw ->
            assertNull(raw, MnnPisaTensorInfoCodec.decode(raw))
        }
    }
}
