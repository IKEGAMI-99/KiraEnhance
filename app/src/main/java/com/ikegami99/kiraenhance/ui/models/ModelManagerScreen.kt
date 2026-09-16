package com.ikegami99.kiraenhance.ui.models

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.ikegami99.kiraenhance.device.SupportTier
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelBackend
import java.util.Locale

internal object ModelCardStatusText {
    fun textFor(
        downloadState: ModelDownloadState,
        wifiOnly: Boolean,
        progressPercent: Int,
        downloadedText: String,
        totalText: String,
    ): String? = ModelDownloadUiReducer.waitingMessage(downloadState, wifiOnly)
        ?: when (downloadState) {
            ModelDownloadState.RUNNING -> "$progressPercent% • $downloadedText / $totalText"
            else -> null
        }
}

@Composable
fun ModelManagerScreen(
    state: ModelManagerUiState,
    onBack: () -> Unit,
    onDownload: (String) -> Unit,
    onDelete: (String) -> Unit,
    onWifiOnlyChanged: (Boolean) -> Unit,
    onOpenLicense: (String) -> Unit,
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
                        text = "モデル管理",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "必要なAIだけ端末へ保存します",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                TextButton(onClick = onBack) {
                    Text("← 戻る")
                }
            }
        }

        state.errorMessage?.let { error ->
            item {
                MessageSurface(
                    message = error,
                    isWarning = true,
                )
            }
        }

        item {
            DeviceCard(state)
        }

        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.large,
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainer,
                ),
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 18.dp, vertical = 14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(2.dp),
                    ) {
                        Text(
                            text = "Wi‑Fiのみでモデルをダウンロード",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = "大きなモデルでモバイル通信を消費しないようにします",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(
                        checked = state.wifiOnly,
                        onCheckedChange = onWifiOnlyChanged,
                    )
                }
            }
        }

        item {
            Text(
                text = "AI MODELS",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }

        items(
            items = state.cards,
            key = { it.model.id },
        ) { card ->
            ModelCard(
                card = card,
                wifiOnly = state.wifiOnly,
                onDownload = { onDownload(card.model.id) },
                onDelete = { onDelete(card.model.id) },
                onOpenLicense = { onOpenLicense(card.model.licenseUrl) },
            )
        }

        item {
            Text(
                text = "KiraEnhance 0.1.0-alpha01 • モデル本体はAPKに含まれません",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(vertical = 8.dp),
            )
        }
    }
}

@Composable
private fun DeviceCard(state: ModelManagerUiState) {
    val capabilities = state.deviceCapabilities
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.large,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
        ),
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "この端末",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            if (capabilities == null) {
                Text(
                    text = "端末性能を確認できませんでした",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(2.dp),
                    ) {
                        Text(
                            text = capabilities.deviceName,
                            style = MaterialTheme.typography.bodyLarge,
                            fontWeight = FontWeight.Medium,
                        )
                        Text(
                            text = "RAM ${formatRam(capabilities.totalRamMb)} • Vulkan ${capabilities.vulkanVersion} • Android API ${capabilities.androidApi}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    TierBadge(capabilities.supportTier)
                }
                Text(
                    text = "空き容量 ${formatStorage(capabilities.availableStorageMb)}",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun TierBadge(tier: SupportTier) {
    val label = when (tier) {
        SupportTier.RECOMMENDED -> "推奨"
        SupportTier.SUPPORTED -> "対応"
        SupportTier.NOT_RECOMMENDED -> "非推奨"
        SupportTier.UNSUPPORTED -> "非対応"
    }
    Surface(
        shape = MaterialTheme.shapes.small,
        color = when (tier) {
            SupportTier.RECOMMENDED,
            SupportTier.SUPPORTED,
            -> MaterialTheme.colorScheme.primaryContainer

            SupportTier.NOT_RECOMMENDED -> MaterialTheme.colorScheme.tertiaryContainer
            SupportTier.UNSUPPORTED -> MaterialTheme.colorScheme.errorContainer
        },
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
        )
    }
}

@Composable
private fun ModelCard(
    card: ModelCardUiState,
    wifiOnly: Boolean,
    onDownload: () -> Unit,
    onDelete: () -> Unit,
    onOpenLicense: () -> Unit,
) {
    val model = card.model
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.large,
        colors = CardDefaults.cardColors(
            containerColor = if (model.mode == EnhancementMode.BALANCED) {
                MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.38f)
            } else {
                MaterialTheme.colorScheme.surfaceContainer
            },
        ),
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.Top,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        text = modeLabel(model.mode),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = model.displayName,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.Medium,
                    )
                }
                if (model.community) {
                    Surface(
                        shape = MaterialTheme.shapes.small,
                        color = MaterialTheme.colorScheme.tertiaryContainer,
                    ) {
                        Text(
                            text = "Community",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 9.dp, vertical = 5.dp),
                        )
                    }
                }
            }

            Text(
                text = model.description,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)

            Text(
                text = buildString {
                    append(formatBytes(model.fileSizeBytes))
                    append(" • RAM目安 ")
                    append(formatRam(model.estimatedRamMb))
                    append(" • ")
                    append(backendLabel(model.backend))
                    append(" • ")
                    append(model.supportedScales.joinToString(" / ") { "${it}x" })
                },
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "v${model.version}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                TextButton(onClick = onOpenLicense) {
                    Text("License: ${model.licenseName}")
                }
            }

            card.warningMessage?.let { warning ->
                MessageSurface(message = warning, isWarning = true)
            }

            if (!card.downloadAvailable && !card.installed) {
                MessageSurface(
                    message = "モデルWeightの配布URLは準備中です。推論ランタイム検証後に有効化します。",
                    isWarning = false,
                )
            }

            if (card.downloadState in setOf(
                    ModelDownloadState.QUEUED,
                    ModelDownloadState.BLOCKED,
                    ModelDownloadState.RUNNING,
                )
            ) {
                val progress = if (card.totalBytes > 0L) {
                    (card.bytesDownloaded.toFloat() / card.totalBytes.toFloat()).coerceIn(0f, 1f)
                } else {
                    0f
                }
                val progressPercent = (progress * 100).toInt()
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier.fillMaxWidth(),
                )
                ModelCardStatusText.textFor(
                    downloadState = card.downloadState,
                    wifiOnly = wifiOnly,
                    progressPercent = progressPercent,
                    downloadedText = formatBytes(card.bytesDownloaded),
                    totalText = formatBytes(card.totalBytes),
                )?.let { statusText ->
                    Text(
                        text = statusText,
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            card.errorMessage?.let { error ->
                Text(
                    text = error,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }

            Spacer(modifier = Modifier.height(2.dp))

            val downloadBusy = card.downloadState in setOf(
                ModelDownloadState.QUEUED,
                ModelDownloadState.BLOCKED,
                ModelDownloadState.RUNNING,
            )
            if (card.installed) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    OutlinedButton(
                        onClick = onDelete,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("削除")
                    }
                    Button(
                        onClick = onDownload,
                        enabled = card.downloadAvailable && !downloadBusy,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("再インストール")
                    }
                }
            } else {
                Button(
                    onClick = onDownload,
                    enabled = card.downloadAvailable && !downloadBusy,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(if (card.downloadAvailable) "ダウンロード" else "配布準備中")
                }
            }
        }
    }
}

@Composable
private fun MessageSurface(
    message: String,
    isWarning: Boolean,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        color = if (isWarning) {
            MaterialTheme.colorScheme.tertiaryContainer
        } else {
            MaterialTheme.colorScheme.surfaceContainerHighest
        },
    ) {
        Text(
            text = message,
            style = MaterialTheme.typography.bodySmall,
            color = if (isWarning) {
                MaterialTheme.colorScheme.onTertiaryContainer
            } else {
                MaterialTheme.colorScheme.onSurfaceVariant
            },
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
        )
    }
}

private fun modeLabel(mode: EnhancementMode): String = when (mode) {
    EnhancementMode.FIDELITY -> "忠実"
    EnhancementMode.BALANCED -> "おすすめ"
    EnhancementMode.DETAIL -> "高精細"
    EnhancementMode.ULTRASHARP -> "UltraSharp"
}

private fun backendLabel(backend: ModelBackend): String = when (backend) {
    ModelBackend.MNN -> "MNN"
    ModelBackend.NCNN -> "ncnn / Vulkan"
}

private fun formatRam(megabytes: Long): String = if (megabytes >= 1024L) {
    String.format(Locale.US, "%.1f GB", megabytes / 1024.0)
} else {
    "$megabytes MB"
}

private fun formatStorage(megabytes: Long): String = if (megabytes >= 1024L) {
    String.format(Locale.US, "%.1f GB", megabytes / 1024.0)
} else {
    "$megabytes MB"
}

private fun formatBytes(bytes: Long): String = when {
    bytes >= 1024L * 1024L * 1024L -> String.format(Locale.US, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
    bytes >= 1024L * 1024L -> String.format(Locale.US, "%.1f MB", bytes / (1024.0 * 1024.0))
    bytes >= 1024L -> String.format(Locale.US, "%.1f KB", bytes / 1024.0)
    else -> "$bytes B"
}
