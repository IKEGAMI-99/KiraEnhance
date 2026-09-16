package com.ikegami99.kiraenhance.ui.smoke

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
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


data class NcnnSmokeScreenState(
    val modelInstalled: Boolean,
    val status: String,
    val imageSelected: Boolean = false,
    val isRunning: Boolean = false,
    val inputDescription: String? = null,
    val outputDescription: String? = null,
    val runtimeDetails: String? = null,
    val elapsedMs: Long? = null,
)

@Composable
fun NcnnSmokeContent(
    state: NcnnSmokeScreenState,
    onBack: () -> Unit,
    onSelectImage: () -> Unit,
    onRun: () -> Unit,
    onOpenModelManager: () -> Unit,
    modifier: Modifier = Modifier,
    inputPreview: ImageBitmap? = null,
    outputPreview: ImageBitmap? = null,
) {
    LazyColumn(
        modifier = modifier,
        contentPadding = PaddingValues(horizontal = 20.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                OutlinedButton(onClick = onBack) {
                    Text("戻る")
                }
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        text = "UltraSharp 実機テスト",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "256px以下の中央クロップを4xで処理します",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        item {
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.large,
                tonalElevation = 2.dp,
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Text(
                        text = "STATUS",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = state.status,
                        style = MaterialTheme.typography.bodyLarge,
                    )
                    if (state.isRunning) {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    }
                    state.runtimeDetails?.let { details ->
                        Text(
                            text = details,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    state.inputDescription?.let { description ->
                        Text(
                            text = "入力: $description",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    state.outputDescription?.let { description ->
                        Text(
                            text = "出力: $description",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    state.elapsedMs?.let { elapsed ->
                        Text(
                            text = "処理時間: ${elapsed} ms",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
        }

        inputPreview?.let { preview ->
            item {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large,
                    tonalElevation = 1.dp,
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text("入力クロップ", style = MaterialTheme.typography.titleSmall)
                        Image(
                            bitmap = preview,
                            contentDescription = "入力プレビュー",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(1f),
                        )
                    }
                }
            }
        }

        outputPreview?.let { preview ->
            item {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large,
                    tonalElevation = 1.dp,
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text("4x出力", style = MaterialTheme.typography.titleSmall)
                        Image(
                            bitmap = preview,
                            contentDescription = "出力プレビュー",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(1f),
                        )
                    }
                }
            }
        }

        item {
            if (!state.modelInstalled) {
                Button(
                    onClick = onOpenModelManager,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("モデル管理を開く")
                }
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedButton(
                        onClick = onSelectImage,
                        enabled = !state.isRunning,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text("画像を選ぶ")
                    }
                    Button(
                        onClick = onRun,
                        enabled = state.imageSelected && !state.isRunning,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text("4xで実行")
                    }
                }
            }
        }

        item {
            Text(
                text = "開発用Smoke Testです。画像は外部へ送信されません。",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
