package com.ikegami99.kiraenhance

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import org.junit.Rule
import org.junit.Test

class KiraEnhanceAppTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun ultraSharpSmokeEntryNavigatesToSmokeRoute() {
        composeRule.setContent {
            KiraEnhanceApp()
        }

        composeRule.onNodeWithText("UltraSharp 実機テスト").performClick()
        composeRule.onNodeWithText("UltraSharp 実機テスト").assertIsDisplayed()
        composeRule.onNodeWithText("開発用Smoke Testです。画像は外部へ送信されません。")
            .assertIsDisplayed()
    }
}
