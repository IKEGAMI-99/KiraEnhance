package com.ikegami99.kiraenhance.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

private data class ModePreview(
    val title: String,
    val description: String,
    val hint: String,
)

private val modePreviews = listOf(
    ModePreview("忠実", "元画像の輪郭や衣装を崩しにくい", "速い・変化少なめ"),
    ModePreview("おすすめ", "自然さと細部の復元をバランス", "標準・迷ったらこれ"),
    ModePreview("高精細", "髪や布地などの細部を強めに復元", "高品質・やや重い"),
    ModePreview("UltraSharp", "シャープで見慣れた質感を狙う", "Community model"),
)

@Composable
fun HomeScreen(
    onOpenModels: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Scaffold(modifier = modifier) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            contentPadding = PaddingValues(horizontal = 20.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            item {
                Text(
                    text = "KiraEnhance",
                    style = MaterialTheme.typography.displaySmall,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = "ゲームスクショを、端末の中だけで高画質化。",
                    modifier = Modifier.padding(top = 8.dp),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodyLarge,
                )
            }

            item {
                Button(
                    onClick = {},
                    enabled = false,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(58.dp),
                ) {
                    Text("画像を選ぶ")
                }
                Text(
                    text = "画像処理エンジンは次のマイルストーンで有効になります",
                    modifier = Modifier.padding(top = 8.dp),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        text = "モード",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.SemiBold,
                    )
                    OutlinedButton(onClick = onOpenModels) {
                        Text("モデル管理")
                    }
                }
            }

            items(modePreviews.size) { index ->
                ModeCard(modePreviews[index])
            }

            item { Spacer(Modifier.height(8.dp)) }
        }
    }
}

@Composable
private fun ModeCard(mode: ModePreview) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.72f),
        ),
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Text(
                text = mode.title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = mode.description,
                modifier = Modifier.padding(top = 6.dp),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Surface(
                modifier = Modifier.padding(top = 12.dp),
                shape = MaterialTheme.shapes.small,
                color = MaterialTheme.colorScheme.primaryContainer,
            ) {
                Text(
                    text = mode.hint,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    style = MaterialTheme.typography.labelMedium,
                )
            }
        }
    }
}
