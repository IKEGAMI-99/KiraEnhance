package com.ikegami99.kiraenhance.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.ikegami99.kiraenhance.model.EnhancementMode

private data class ModePreview(
    val icon: String,
    val title: String,
    val subtitle: String,
    val mode: EnhancementMode,
    val enabled: Boolean,
)

private val modes = listOf(
    ModePreview(
        icon = "🪡",
        title = "忠実",
        subtitle = "元画像の形や細かな模様をできるだけ維持",
        mode = EnhancementMode.FIDELITY,
        enabled = false,
    ),
    ModePreview(
        icon = "✨",
        title = "おすすめ",
        subtitle = "PiSA-SRで忠実度とAIディテールのバランスを重視",
        mode = EnhancementMode.BALANCED,
        enabled = true,
    ),
    ModePreview(
        icon = "💎",
        title = "高精細",
        subtitle = "髪や衣装の細かな質感をより鮮明に",
        mode = EnhancementMode.DETAIL,
        enabled = false,
    ),
    ModePreview(
        icon = "🔮",
        title = "UltraSharp",
        subtitle = "使い慣れた4x-UltraSharpをCommunity Modelとして利用",
        mode = EnhancementMode.ULTRASHARP,
        enabled = true,
    ),
)

@Composable
fun HomeScreen(
    onOpenModelManager: () -> Unit,
    onOpenSmokeTest: () -> Unit,
    onOpenAppUpdate: () -> Unit = {},
    onSaveLogs: () -> Unit = {},
    onStartEnhance: (EnhancementMode) -> Unit = {},
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier,
        contentPadding = PaddingValues(horizontal = 20.dp, vertical = 32.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(
                    text = "KiraEnhance",
                    style = MaterialTheme.typography.displaySmall,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "ゲームスクショを、端末の中だけで美しく。",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        item {
            Surface(
                shape = MaterialTheme.shapes.large,
                tonalElevation = 2.dp,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text(
                        text = "画像を選んで高画質化",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = "画像は外部へ送信しません。AI処理は完全オンデバイスです。",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Button(
                        onClick = { onStartEnhance(EnhancementMode.BALANCED) },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text("＋ 画像を選ぶ")
                    }
                    Text(
                        text = "おすすめはPiSA-SR、UltraSharpは検証済み4xエンジンです",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        item {
            Text(
                text = "AI MODE",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }

        items(modes) { mode ->
            Surface(
                onClick = { onStartEnhance(mode.mode) },
                enabled = mode.enabled,
                shape = MaterialTheme.shapes.medium,
                tonalElevation = if (mode.title == "おすすめ") 4.dp else 1.dp,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Row(
                    modifier = Modifier.padding(18.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(14.dp),
                ) {
                    Text(
                        text = mode.icon,
                        style = MaterialTheme.typography.headlineMedium,
                    )
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(3.dp),
                    ) {
                        Text(
                            text = mode.title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = mode.subtitle,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        if (!mode.enabled) {
                            Text(
                                text = "準備中",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }

        item {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(
                    onClick = onOpenModelManager,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("モデル管理")
                }
                OutlinedButton(
                    onClick = onOpenAppUpdate,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("アプリ更新")
                }
                OutlinedButton(
                    onClick = onSaveLogs,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("ログを保存")
                }
                OutlinedButton(
                    onClick = onOpenSmokeTest,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("UltraSharp 実機テスト")
                }
            }
        }
    }
}
