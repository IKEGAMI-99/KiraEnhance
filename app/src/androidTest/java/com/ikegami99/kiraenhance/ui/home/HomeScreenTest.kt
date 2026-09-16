package com.ikegami99.kiraenhance.ui.home

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
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
}
