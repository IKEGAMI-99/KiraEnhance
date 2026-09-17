package com.ikegami99.kiraenhance.ui.home

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.ikegami99.kiraenhance.model.EnhancementMode
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class HomeScreenTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun showsUltraSharpSmokeTestEntryAndLogSaveAction() {
        composeRule.setContent {
            HomeScreen(
                onOpenModelManager = {},
                onOpenSmokeTest = {},
                onSaveLogs = {},
            )
        }

        composeRule.onNodeWithText("UltraSharp 実機テスト").assertIsDisplayed()
        composeRule.onNodeWithText("ログを保存").assertIsDisplayed()
    }

    @Test
    fun primaryImageActionStartsRecommendedBalancedMode() {
        var selectedMode: EnhancementMode? = null
        composeRule.setContent {
            HomeScreen(
                onOpenModelManager = {},
                onOpenSmokeTest = {},
                onStartEnhance = { selectedMode = it },
            )
        }

        composeRule.onNodeWithText("＋ 画像を選ぶ")
            .assertIsEnabled()
            .performClick()

        assertEquals(EnhancementMode.BALANCED, selectedMode)
    }

    @Test
    fun recommendedModeStartsBalancedEnhancement() {
        var selectedMode: EnhancementMode? = null
        composeRule.setContent {
            HomeScreen(
                onOpenModelManager = {},
                onOpenSmokeTest = {},
                onStartEnhance = { selectedMode = it },
            )
        }

        composeRule.onNodeWithText("おすすめ")
            .assertIsEnabled()
            .performClick()

        assertEquals(EnhancementMode.BALANCED, selectedMode)
    }

    @Test
    fun ultraSharpModeStartsUltraSharpEnhancement() {
        var selectedMode: EnhancementMode? = null
        composeRule.setContent {
            HomeScreen(
                onOpenModelManager = {},
                onOpenSmokeTest = {},
                onStartEnhance = { selectedMode = it },
            )
        }

        composeRule.onNodeWithText("UltraSharp")
            .assertIsEnabled()
            .performClick()

        assertEquals(EnhancementMode.ULTRASHARP, selectedMode)
    }

    @Test
    fun fidelityAndDetailModesAreDisabledUntilTheirEnginesExist() {
        composeRule.setContent {
            HomeScreen(
                onOpenModelManager = {},
                onOpenSmokeTest = {},
            )
        }

        composeRule.onNodeWithText("忠実").assertIsNotEnabled()
        composeRule.onNodeWithText("高精細").assertIsNotEnabled()
    }
}
