package com.ikegami99.kiraenhance.ui.enhance

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import org.junit.Rule
import org.junit.Test

class EnhanceScreenTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun readyStateShowsEnhanceAction() {
        composeRule.setContent {
            EnhanceScreen(
                state = EnhanceUiState(
                    stage = EnhanceStage.READY,
                    sourceWidth = 100,
                    sourceHeight = 200,
                    targetWidth = 400,
                    targetHeight = 800,
                ),
                sourcePreview = null,
                resultPreview = null,
                onBack = {},
                onSelectImage = {},
                onEnhance = {},
                onCancel = {},
                onSave = {},
                onRunAgain = {},
            )
        }

        composeRule.onNodeWithText("高画質化する").assertIsDisplayed()
    }

    @Test
    fun processingStateShowsCancelAction() {
        composeRule.setContent {
            EnhanceScreen(
                state = EnhanceUiState(
                    stage = EnhanceStage.PROCESSING,
                    sourceWidth = 100,
                    sourceHeight = 200,
                    targetWidth = 400,
                    targetHeight = 800,
                    progressFraction = 0.5f,
                    progressText = "2 / 4 tiles",
                ),
                sourcePreview = null,
                resultPreview = null,
                onBack = {},
                onSelectImage = {},
                onEnhance = {},
                onCancel = {},
                onSave = {},
                onRunAgain = {},
            )
        }

        composeRule.onNodeWithText("キャンセル").assertIsDisplayed()
    }
}
