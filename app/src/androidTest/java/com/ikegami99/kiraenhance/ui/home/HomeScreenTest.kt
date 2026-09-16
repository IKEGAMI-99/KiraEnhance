package com.ikegami99.kiraenhance.ui.home

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import org.junit.Assert.assertTrue
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
    fun primaryImageActionStartsProductionEnhanceFlow() {
        var started = false
        composeRule.setContent {
            HomeScreen(
                onOpenModelManager = {},
                onOpenSmokeTest = {},
                onStartEnhance = { started = true },
            )
        }

        composeRule.onNodeWithText("＋ 画像を選ぶ")
            .assertIsEnabled()
            .performClick()

        assertTrue(started)
    }
}
