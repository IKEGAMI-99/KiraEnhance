package com.ikegami99.kiraenhance.ui.smoke

import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeImage
import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeInferenceRunner
import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeRunResult
import com.ikegami99.kiraenhance.model.ModelDescriptor

data class NcnnSmokePresentationResult(
    val state: NcnnSmokeScreenState,
    val output: NcnnSmokeImage?,
)

class NcnnSmokeInferencePresenter(
    private val runner: NcnnSmokeInferenceRunner,
    private val nowMs: () -> Long = System::currentTimeMillis,
) {
    fun run(
        model: ModelDescriptor,
        artifactPath: (String) -> String,
        image: NcnnSmokeImage,
    ): NcnnSmokePresentationResult {
        val startedAt = nowMs()
        val result = runner.run(
            model = model,
            artifactPath = artifactPath,
            image = image,
        )
        val elapsedMs = (nowMs() - startedAt).coerceAtLeast(0L)

        return when (result) {
            is NcnnSmokeRunResult.Success -> {
                val output = NcnnSmokeImage(
                    width = result.width,
                    height = result.height,
                    argbPixels = result.argbPixels,
                )
                NcnnSmokePresentationResult(
                    state = NcnnSmokeScreenState(
                        modelInstalled = true,
                        status = "4x処理が完了しました",
                        imageSelected = true,
                        inputDescription = "${image.width} × ${image.height}",
                        outputDescription = "${output.width} × ${output.height}",
                        runtimeDetails = if (result.usedGpu) "Vulkan GPU" else "CPU",
                        elapsedMs = elapsedMs,
                    ),
                    output = output,
                )
            }

            is NcnnSmokeRunResult.Failed -> NcnnSmokePresentationResult(
                state = NcnnSmokeScreenState(
                    modelInstalled = true,
                    status = "処理に失敗しました: ${result.error.message}",
                    imageSelected = true,
                    inputDescription = "${image.width} × ${image.height}",
                    elapsedMs = elapsedMs,
                ),
                output = null,
            )
        }
    }
}
