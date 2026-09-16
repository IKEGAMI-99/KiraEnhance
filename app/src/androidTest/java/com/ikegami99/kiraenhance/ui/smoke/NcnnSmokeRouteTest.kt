package com.ikegami99.kiraenhance.ui.smoke

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import org.junit.Rule
import org.junit.Test

class NcnnSmokeRouteTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun routeRendersSmokeScreen() {
        composeRule.setContent {
            NcnnSmokeRoute(
                onBack = {},
                onOpenModelManager = {},
            )
        }

        composeRule.onNodeWithText("UltraSharp 実機テスト").assertIsDisplayed()
    }
}
