package com.ikegami99.kiraenhance.ui.enhance

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun EnhanceScreen(
    state: EnhanceUiState,
    sourcePreview: ImageBitmap?,
    resultPreview: ImageBitmap?,
    onBack: () -> Unit,
    onSelectImage: () -> Unit,
    onEnhance: () -> Unit,
    onCancel: () -> Unit,
    onSave: () -> Unit,
    onRunAgain: () -> Unit,
    onOpenModelManager: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            OutlinedButton(onClick = onBack) {
                Text("戻る")
            }
            Text(
                text = "UltraSharp 4x",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )
        }

        when (state.stage) {
            EnhanceStage.SELECTING -> SelectingContent(state, onSelectImage)
            EnhanceStage.READY -> ReadyContent(state, sourcePreview, onSelectImage, onEnhance)
            EnhanceStage.PROCESSING -> ProcessingContent(state, sourcePreview, onCancel)
            EnhanceStage.COMPLETED,
            EnhanceStage.SAVED,
            -> CompletedContent(state, sourcePreview, resultPreview, onSave, onRunAgain, onSelectImage)
            EnhanceStage.ERROR -> ErrorContent(state, sourcePreview, onRunAgain, onSelectImage, onOpenModelManager)
        }
    }
}

@Composable
private fun SelectingContent(
    state: EnhanceUiState,
    onSelectImage: () -> Unit,
) {
    MessageCard(title = "画像を選択", message = state.status)
    Button(
        onClick = onSelectImage,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text("画像を選ぶ")
    }
}

@Composable
private fun ReadyContent(
    state: EnhanceUiState,
    sourcePreview: ImageBitmap?,
    onSelectImage: () -> Unit,
    onEnhance: () -> Unit,
) {
    PreviewCard(sourcePreview, "選択した画像")
    ResolutionCard(state)
    if (state.status == "処理をキャンセルしました") {
        MessageCard("状態", state.status)
    }
    MessageCard(
        title = "4x-UltraSharp",
        message = "端末内のncnn/Vulkanで4倍に高画質化します。画像データは外部へ送信しません。",
    )
    Button(
        onClick = onEnhance,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text("高画質化する")
    }
    OutlinedButton(
        onClick = onSelectImage,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text("別の画像を選ぶ")
    }
}

@Composable
private fun ProcessingContent(
    state: EnhanceUiState,
    sourcePreview: ImageBitmap?,
    onCancel: () -> Unit,
) {
    PreviewCard(sourcePreview, "処理中の画像")
    Text(
        text = state.status,
        style = MaterialTheme.typography.titleMedium,
        fontWeight = FontWeight.SemiBold,
    )
    LinearProgressIndicator(
        progress = { state.progressFraction.coerceIn(0f, 1f) },
        modifier = Modifier.fillMaxWidth(),
    )
    Text(
        text = state.progressText ?: "タイル計画を準備中...",
        style = MaterialTheme.typography.bodyMedium,
    )
    Text(
        text = "経過 ${(state.elapsedMs / 100) / 10.0} 秒",
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
    ResolutionCard(state)
    OutlinedButton(
        onClick = onCancel,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text("キャンセル")
    }
}

@Composable
private fun CompletedContent(
    state: EnhanceUiState,
    sourcePreview: ImageBitmap?,
    resultPreview: ImageBitmap?,
    onSave: () -> Unit,
    onRunAgain: () -> Unit,
    onSelectImage: () -> Unit,
) {
    if (sourcePreview != null && resultPreview != null) {
        Text(
            text = "左右にスライドして比較 / ピンチで拡大",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        CompareViewer(
            original = sourcePreview,
            enhanced = resultPreview,
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(0.75f),
        )
    } else {
        MessageCard("比較プレビュー", "プレビュー画像を表示できませんでした。")
    }

    ResolutionCard(state)
    Text(
        text = "処理時間 ${(state.elapsedMs / 100) / 10.0} 秒",
        style = MaterialTheme.typography.bodyMedium,
    )
    if (state.stage == EnhanceStage.SAVED && state.savedLocation != null) {
        MessageCard("保存完了", state.savedLocation)
    } else if (state.status.startsWith("保存に失敗しました")) {
        MessageCard("保存エラー", state.status)
    }
    Button(
        onClick = onSave,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text(if (state.stage == EnhanceStage.SAVED) "もう一度保存" else "保存する")
    }
    OutlinedButton(
        onClick = onRunAgain,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text("もう一度処理")
    }
    OutlinedButton(
        onClick = onSelectImage,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text("別の画像を選ぶ")
    }
}

@Composable
private fun ErrorContent(
    state: EnhanceUiState,
    sourcePreview: ImageBitmap?,
    onRunAgain: () -> Unit,
    onSelectImage: () -> Unit,
    onOpenModelManager: () -> Unit,
) {
    PreviewCard(sourcePreview, "元画像")
    MessageCard("処理できませんでした", state.status)
    if (sourcePreview != null) {
        Button(
            onClick = onRunAgain,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("もう一度試す")
        }
    }
    OutlinedButton(
        onClick = onSelectImage,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text("別の画像を選ぶ")
    }
    OutlinedButton(
        onClick = onOpenModelManager,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text("モデル管理を開く")
    }
}

@Composable
private fun PreviewCard(image: ImageBitmap?, contentDescription: String) {
    if (image == null) return
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.large,
        tonalElevation = 2.dp,
    ) {
        Image(
            bitmap = image,
            contentDescription = contentDescription,
            contentScale = ContentScale.Fit,
            modifier = Modifier
                .fillMaxWidth()
                .height(260.dp)
                .padding(8.dp),
        )
    }
}

@Composable
private fun ResolutionCard(state: EnhanceUiState) {
    val source = if (state.sourceWidth != null && state.sourceHeight != null) {
        "${state.sourceWidth} × ${state.sourceHeight}"
    } else {
        "-"
    }
    val target = if (state.targetWidth != null && state.targetHeight != null) {
        "${state.targetWidth} × ${state.targetHeight}"
    } else {
        "-"
    }
    MessageCard("解像度", "$source  →  $target")
}

@Composable
private fun MessageCard(title: String, message: String) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.large,
        tonalElevation = 2.dp,
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
