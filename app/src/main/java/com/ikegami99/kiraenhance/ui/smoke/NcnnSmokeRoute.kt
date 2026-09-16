package com.ikegami99.kiraenhance.ui.smoke

import android.content.Context
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import com.ikegami99.kiraenhance.model.InstalledModelStore
import com.ikegami99.kiraenhance.model.ModelManifestParser

@Composable
fun NcnnSmokeRoute(
    onBack: () -> Unit,
    onOpenModelManager: () -> Unit,
) {
    val context = LocalContext.current
    val state = remember(context.applicationContext) {
        createInitialState(context.applicationContext)
    }

    NcnnSmokeContent(
        state = state,
        onBack = onBack,
        onSelectImage = {},
        onRun = {},
        onOpenModelManager = onOpenModelManager,
    )
}

private fun createInitialState(context: Context): NcnnSmokeScreenState {
    val model = runCatching {
        val json = context.assets.open("model-manifest.json")
            .bufferedReader()
            .use { it.readText() }
        ModelManifestParser()
            .parse(json)
            .models
            .first { it.id == ULTRASHARP_MODEL_ID }
    }.getOrElse { error ->
        return NcnnSmokeScreenState(
            modelInstalled = false,
            status = "4x-UltraSharpモデル情報を読み込めません: ${error.message ?: "unknown error"}",
        )
    }

    val installed = InstalledModelStore(context).isInstalled(model)
    return NcnnSmokeScreenState(
        modelInstalled = installed,
        status = if (installed) {
            "画像を選択してください"
        } else {
            "4x-UltraSharpモデルが未インストールです"
        },
    )
}

private const val ULTRASHARP_MODEL_ID = "ultrasharp"
