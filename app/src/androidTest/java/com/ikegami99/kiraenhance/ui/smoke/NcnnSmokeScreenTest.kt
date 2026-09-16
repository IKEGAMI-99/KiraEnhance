package com.ikegami99.kiraenhance.ui.smoke

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import org.junit.Rule
import org.junit.Test

class NcnnSmokeScreenTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun missingModelShowsModelManagerAction() {
        composeRule.setContent {
            NcnnSmokeContent(
                state = NcnnSmokeScreenState(
                    modelInstalled = false,
                    status = "4x-UltraSharpモデルが未インストールです",
                ),
                onBack = {},
                onSelectImage = {},
                onRun = {},
                onOpenModelManager = {},
            )
        }

        composeRule.onNodeWithText("UltraSharp 実機テスト").assertIsDisplayed()
        composeRule.onNodeWithText("4x-UltraSharpモデルが未インストールです").assertIsDisplayed()
        composeRule.onNodeWithText("モデル管理を開く").assertIsDisplayed()
    }

    @Test
    fun installedModelShowsImagePickerAction() {
        composeRule.setContent {
            NcnnSmokeContent(
                state = NcnnSmokeScreenState(
                    modelInstalled = true,
                    status = "画像を選択してください",
                ),
                onBack = {},
                onSelectImage = {},
                onRun = {},
                onOpenModelManager = {},
            )
        }

        composeRule.onNodeWithText("画像を選ぶ").assertIsDisplayed()
        composeRule.onNodeWithText("4xで実行").assertIsDisplayed()
    }
}
