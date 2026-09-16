package com.ikegami99.kiraenhance.ui.smoke

import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeImage

data class NcnnSmokeSelectionResult(
    val state: NcnnSmokeScreenState,
    val image: NcnnSmokeImage,
)

object NcnnSmokeSelectionPresenter {
    fun present(image: NcnnSmokeImage): NcnnSmokeSelectionResult =
        NcnnSmokeSelectionResult(
            state = NcnnSmokeScreenState(
                modelInstalled = true,
                status = "画像を選択しました",
                imageSelected = true,
                inputDescription = "${image.width} × ${image.height}",
            ),
            image = image,
        )
}
