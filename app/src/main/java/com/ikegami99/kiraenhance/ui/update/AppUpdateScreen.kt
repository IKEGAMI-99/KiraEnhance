package com.ikegami99.kiraenhance.ui.update

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.util.Locale

@Composable
fun AppUpdateScreen(
    state: AppUpdateUiState,
    onBack: () -> Unit,
    onCheck: () -> Unit,
    onDownload: () -> Unit,
    onInstall: () -> Unit,
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier,
        contentPadding = PaddingValues(horizontal = 20.dp, vertical = 20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        text = "アプリ更新",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "現在 v${state.currentVersionName}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                TextButton(onClick = onBack) {
                    Text("← 戻る")
                }
            }
        }

        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.large,
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainer,
                ),
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    UpdateStatus(state)

                    when (state.stage) {
                        AppUpdateStage.CHECKING -> {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                            Text(
                                text = "GitHub Releasesを確認しています…",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }

                        AppUpdateStage.DOWNLOADING -> DownloadProgress(state)
                        else -> UpdatePrimaryAction(
                            stage = state.stage,
                            onCheck = onCheck,
                            onDownload = onDownload,
                            onInstall = onInstall,
                        )
                    }
                }
            }
        }

        state.updateInfo?.let { info ->
            item {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Column(
                        modifier = Modifier.padding(18.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(
                            text = "v${info.versionName}",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = "更新内容",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = info.releaseNotes.ifBlank { "リリースノートはありません" },
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
            }
        }

        item {
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.medium,
                color = MaterialTheme.colorScheme.surfaceContainerHighest,
            ) {
                Text(
                    text = "更新APKはGitHub ReleasesからHTTPSで取得し、SHA-256が一致した場合だけインストール可能にします。",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(14.dp),
                )
            }
        }

        if (state.stage == AppUpdateStage.READY_TO_INSTALL) {
            item {
                OutlinedButton(
                    onClick = onCheck,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("更新情報を再確認")
                }
            }
        }
    }
}

@Composable
private fun UpdatePrimaryAction(
    stage: AppUpdateStage,
    onCheck: () -> Unit,
    onDownload: () -> Unit,
    onInstall: () -> Unit,
) {
    when (AppUpdateUiReducer.primaryActionFor(stage)) {
        AppUpdatePrimaryAction.NONE -> Unit
        AppUpdatePrimaryAction.CHECK -> Button(
            onClick = onCheck,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (stage == AppUpdateStage.IDLE) "更新を確認" else "もう一度確認")
        }

        AppUpdatePrimaryAction.DOWNLOAD -> Button(
            onClick = onDownload,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("アップデートをダウンロード")
        }

        AppUpdatePrimaryAction.INSTALL -> Button(
            onClick = onInstall,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("インストール")
        }
    }
}

@Composable
private fun UpdateStatus(state: AppUpdateUiState) {
    val title = when (state.stage) {
        AppUpdateStage.IDLE -> "更新を確認できます"
        AppUpdateStage.CHECKING -> "確認中"
        AppUpdateStage.UP_TO_DATE -> "最新バージョンです"
        AppUpdateStage.AVAILABLE -> "v${state.latestVersionName ?: "?"} が利用できます"
        AppUpdateStage.DOWNLOADING -> "更新をダウンロード中"
        AppUpdateStage.READY_TO_INSTALL -> "インストール準備完了"
        AppUpdateStage.ERROR -> "更新処理でエラーが発生しました"
    }

    Text(
        text = title,
        style = MaterialTheme.typography.titleMedium,
        fontWeight = FontWeight.SemiBold,
    )

    state.statusMessage?.let { message ->
        Text(
            text = message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }

    state.errorMessage?.let { error ->
        Text(
            text = error,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.error,
        )
    }
}

@Composable
private fun DownloadProgress(state: AppUpdateUiState) {
    if (state.totalBytes > 0L) {
        val progress = (state.bytesDownloaded.toFloat() / state.totalBytes.toFloat()).coerceIn(0f, 1f)
        LinearProgressIndicator(
            progress = { progress },
            modifier = Modifier.fillMaxWidth(),
        )
        Text(
            text = "${AppUpdateUiReducer.progressPercent(state.bytesDownloaded, state.totalBytes)}% • ${formatBytes(state.bytesDownloaded)} / ${formatBytes(state.totalBytes)}",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    } else {
        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    }
}

private fun formatBytes(bytes: Long): String = when {
    bytes >= 1024L * 1024L -> String.format(Locale.US, "%.1f MB", bytes / (1024.0 * 1024.0))
    bytes >= 1024L -> String.format(Locale.US, "%.1f KB", bytes / 1024.0)
    else -> "$bytes B"
}
