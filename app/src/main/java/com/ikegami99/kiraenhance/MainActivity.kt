package com.ikegami99.kiraenhance

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.ikegami99.kiraenhance.ui.theme.KiraEnhanceTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            KiraEnhanceTheme {
                KiraEnhanceApp()
            }
        }
    }
}
