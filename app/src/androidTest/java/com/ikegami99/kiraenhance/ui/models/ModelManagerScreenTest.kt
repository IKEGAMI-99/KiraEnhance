package com.ikegami99.kiraenhance.ui.models

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.ikegami99.kiraenhance.device.DeviceCapabilities
import com.ikegami99.kiraenhance.device.SupportTier
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelArtifactDescriptor
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor
import com.ikegami99.kiraenhance.ui.theme.KiraEnhanceTheme
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ModelManagerScreenTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun rendersAllModesCommunityBadgeAndActions() {
        composeRule.setContent {
            KiraEnhanceTheme {
                ModelManagerScreen(
                    state = fakeState(),
                    onBack = {},
                    onDownload = {},
                    onDelete = {},
                    onWifiOnlyChanged = {},
                    onImportValidation = {},
                    onProbeValidation = {},
                    onOpenLicense = {},
                )
            }
        }

        listOf("忠実", "おすすめ", "高精細", "UltraSharp").forEach { label ->
            composeRule.onNodeWithText(label).assertIsDisplayed()
        }
        composeRule.onNodeWithText("Community").assertIsDisplayed()
        composeRule.onNodeWithText("削除").assertIsDisplayed()
        composeRule.onAllNodesWithText("ダウンロード")[0].assertIsEnabled()
        composeRule.onNodeWithText("この端末では処理が重くなる可能性があります").assertIsDisplayed()
    }

    @Test
    fun explainsQueuedWifiOnlyAndBlockedWaits() {
        composeRule.setContent {
            KiraEnhanceTheme {
                ModelManagerScreen(
                    state = ModelManagerUiState(
                        cards = listOf(
                            card(
                                id = "ultrasharp",
                                mode = EnhancementMode.ULTRASHARP,
                                community = true,
                                downloadState = ModelDownloadState.QUEUED,
                            ),
                            card(
                                id = "pisa-sr",
                                mode = EnhancementMode.BALANCED,
                                downloadState = ModelDownloadState.BLOCKED,
                            ),
                        ),
                        wifiOnly = true,
                    ),
                    onBack = {},
                    onDownload = {},
                    onDelete = {},
                    onWifiOnlyChanged = {},
                    onImportValidation = {},
                    onProbeValidation = {},
                    onOpenLicense = {},
                )
            }
        }

        composeRule.onNodeWithText(
            "Wi‑Fiのみ設定のため、非従量制ネットワークを待っています",
        ).assertIsDisplayed()
        composeRule.onNodeWithText(
            "前提となる処理の完了を待っています",
        ).assertIsDisplayed()
    }

    private fun fakeState(): ModelManagerUiState = ModelManagerUiState(
        cards = listOf(
            card("realesrgan-anime", EnhancementMode.FIDELITY, installed = true),
            card("pisa-sr", EnhancementMode.BALANCED),
            card("hat-s", EnhancementMode.DETAIL),
            card("ultrasharp", EnhancementMode.ULTRASHARP, community = true),
        ),
        deviceCapabilities = DeviceCapabilities(
            totalRamMb = 6_000,
            availableStorageMb = 20_000,
            vulkanVersion = "1.3",
            deviceName = "Test Device",
            androidApi = 36,
            supportTier = SupportTier.NOT_RECOMMENDED,
        ),
        wifiOnly = true,
    )

    private fun card(
        id: String,
        mode: EnhancementMode,
        installed: Boolean = false,
        community: Boolean = false,
        downloadState: ModelDownloadState = ModelDownloadState.IDLE,
    ): ModelCardUiState = ModelCardUiState(
        model = ModelDescriptor(
            id = id,
            displayName = when (mode) {
                EnhancementMode.FIDELITY -> "RealESRGAN Anime candidate"
                EnhancementMode.BALANCED -> "PiSA-SR"
                EnhancementMode.DETAIL -> "HAT-S candidate"
                EnhancementMode.ULTRASHARP -> "4x-UltraSharp"
            },
            mode = mode,
            version = "candidate-1",
            backend = if (mode == EnhancementMode.FIDELITY || mode == EnhancementMode.ULTRASHARP) {
                ModelBackend.NCNN
            } else {
                ModelBackend.MNN
            },
            artifacts = listOf(
                ModelArtifactDescriptor(
                    fileName = if (mode == EnhancementMode.FIDELITY || mode == EnhancementMode.ULTRASHARP) {
                        "model.bin"
                    } else {
                        "model.mnn"
                    },
                    downloadUrl = "https://example.invalid/$id/model.bin",
                    fileSizeBytes = 64L * 1024L * 1024L,
                    sha256 = "a".repeat(64),
                ),
            ),
            supportedScales = listOf(4),
            minAppVersion = "0.1.0-alpha01",
            estimatedRamMb = 4_096,
            licenseName = if (community) "CC BY-NC-SA 4.0" else "Candidate license",
            licenseUrl = "https://example.invalid/license",
            description = "Model description",
            community = community,
        ),
        installed = installed,
        downloadState = downloadState,
        warningMessage = "この端末では処理が重くなる可能性があります",
    )
}
