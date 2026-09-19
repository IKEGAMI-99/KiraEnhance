package com.ikegami99.kiraenhance.inference.mnn

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MnnPisaNativeLoaderInstrumentedTest {
    @Test
    fun invalidMnnGraphsFailCleanly() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.cacheDir, "mnn-loader-invalid-test")
        directory.deleteRecursively()
        check(directory.mkdirs()) { "Unable to create native loader test directory" }

        try {
            val vaeEncoder = File(directory, "vae_encoder.mnn").apply {
                writeBytes(byteArrayOf(1))
            }
            val unet = File(directory, "unet_default.mnn").apply {
                writeBytes(byteArrayOf(2))
            }
            val vaeDecoder = File(directory, "vae_decoder.mnn").apply {
                writeBytes(byteArrayOf(3))
            }
            val emptyPrompt = File(directory, "empty_prompt.fp16").apply {
                writeBytes(byteArrayOf(0, 0))
            }
            val vaeSegmentPack = File(directory, "vae_segments.pack").apply {
                writeBytes(byteArrayOf(4))
            }

            val result = MnnPisaNativeBridge.loadModel(
                vaeEncoderPath = vaeEncoder.absolutePath,
                unetPath = unet.absolutePath,
                vaeDecoderPath = vaeDecoder.absolutePath,
                emptyPromptPath = emptyPrompt.absolutePath,
                vaeSegmentPackPath = vaeSegmentPack.absolutePath,
                preferGpu = true,
            )

            assertEquals(0L, result.handle)
            assertEquals(MnnPisaNativeError.LOAD_FAILED, result.errorCode)
            assertEquals(false, result.gpuEnabled)
        } finally {
            directory.deleteRecursively()
        }
    }
}
