package com.ikegami99.kiraenhance.ui.models

import android.content.Context
import androidx.lifecycle.LiveData
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.ikegami99.kiraenhance.device.DeviceCapabilities
import com.ikegami99.kiraenhance.device.DeviceCapabilityDetector
import com.ikegami99.kiraenhance.device.SupportTier
import com.ikegami99.kiraenhance.download.ModelDownloadManager
import com.ikegami99.kiraenhance.download.ModelDownloadWorker
import com.ikegami99.kiraenhance.model.InstalledModelStore
import com.ikegami99.kiraenhance.model.ModelDescriptor
import com.ikegami99.kiraenhance.model.ModelManifestParser
import java.net.URI
import java.util.UUID
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class ModelDownloadState {
    IDLE,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    FAILED,
}

data class ModelCardUiState(
    val model: ModelDescriptor,
    val installed: Boolean,
    val downloadState: ModelDownloadState = ModelDownloadState.IDLE,
    val bytesDownloaded: Long = 0L,
    val totalBytes: Long = model.fileSizeBytes,
    val warningMessage: String? = null,
    val errorMessage: String? = null,
    val downloadAvailable: Boolean = true,
)

data class ModelManagerUiState(
    val cards: List<ModelCardUiState> = emptyList(),
    val deviceCapabilities: DeviceCapabilities? = null,
    val wifiOnly: Boolean = true,
    val errorMessage: String? = null,
)

class ModelManagerViewModel(
    private val models: List<ModelDescriptor>,
    private val store: InstalledModelStore,
    private val deviceDetector: DeviceCapabilityDetector,
    private val downloadManager: ModelDownloadManager,
    private val workManager: WorkManager,
    initialError: String? = null,
) : ViewModel() {
    private val _state = MutableStateFlow(ModelManagerUiState(errorMessage = initialError))
    val state: StateFlow<ModelManagerUiState> = _state.asStateFlow()

    private val workObservers = mutableMapOf<UUID, Pair<LiveData<WorkInfo?>, Observer<WorkInfo?>>>()

    init {
        refreshCards()
    }

    fun setWifiOnly(enabled: Boolean) {
        _state.value = _state.value.copy(wifiOnly = enabled)
    }

    fun download(modelId: String) {
        val card = _state.value.cards.firstOrNull { it.model.id == modelId } ?: return
        if (!card.downloadAvailable) {
            updateCard(modelId) {
                it.copy(errorMessage = "このモデルは配布URLの準備中です")
            }
            return
        }
        if (card.downloadState == ModelDownloadState.RUNNING || card.downloadState == ModelDownloadState.QUEUED) {
            return
        }

        val requestId = downloadManager.enqueue(card.model, _state.value.wifiOnly)
        updateCard(modelId) {
            it.copy(
                downloadState = ModelDownloadState.QUEUED,
                bytesDownloaded = 0L,
                totalBytes = card.model.fileSizeBytes,
                errorMessage = null,
            )
        }
        observeDownload(card.model, requestId)
    }

    fun delete(modelId: String) {
        val card = _state.value.cards.firstOrNull { it.model.id == modelId } ?: return
        workManager.cancelUniqueWork(ModelDownloadManager.workName(card.model))

        runCatching {
            store.modelFile(card.model.id, card.model.version).parentFile?.deleteRecursively()
            store.partialFile(card.model.id, card.model.version).delete()
        }.onFailure { error ->
            updateCard(modelId) {
                it.copy(errorMessage = error.message ?: "モデルの削除に失敗しました")
            }
            return
        }

        updateCard(modelId) {
            it.copy(
                installed = false,
                downloadState = ModelDownloadState.IDLE,
                bytesDownloaded = 0L,
                errorMessage = null,
            )
        }
    }

    fun reinstall(modelId: String) {
        delete(modelId)
        download(modelId)
    }

    private fun refreshCards() {
        val capabilitiesResult = runCatching { deviceDetector.detect() }
        val capabilities = capabilitiesResult.getOrNull()
        val warning = warningFor(capabilities)
        val detectorError = capabilitiesResult.exceptionOrNull()?.let {
            "端末性能の判定に失敗しました: ${it.message ?: "unknown error"}"
        }

        _state.value = _state.value.copy(
            deviceCapabilities = capabilities,
            errorMessage = _state.value.errorMessage ?: detectorError,
            cards = models.map { model ->
                ModelCardUiState(
                    model = model,
                    installed = store.isInstalled(model.id, model.version),
                    warningMessage = warning,
                    downloadAvailable = hasProductionDownloadUrl(model.downloadUrl),
                )
            },
        )
    }

    private fun observeDownload(model: ModelDescriptor, requestId: UUID) {
        val liveData = workManager.getWorkInfoByIdLiveData(requestId)
        val observer = Observer<WorkInfo?> { info ->
            if (info == null) return@Observer

            val downloaded = info.progress.getLong(ModelDownloadWorker.KEY_BYTES_DOWNLOADED, 0L)
            val total = info.progress.getLong(ModelDownloadWorker.KEY_TOTAL_BYTES, model.fileSizeBytes)
                .takeIf { it > 0L }
                ?: model.fileSizeBytes

            when (info.state) {
                WorkInfo.State.ENQUEUED,
                WorkInfo.State.BLOCKED,
                -> updateCard(model.id) {
                    it.copy(
                        downloadState = ModelDownloadState.QUEUED,
                        bytesDownloaded = downloaded,
                        totalBytes = total,
                    )
                }

                WorkInfo.State.RUNNING -> updateCard(model.id) {
                    it.copy(
                        downloadState = ModelDownloadState.RUNNING,
                        bytesDownloaded = downloaded,
                        totalBytes = total,
                        errorMessage = null,
                    )
                }

                WorkInfo.State.SUCCEEDED -> {
                    updateCard(model.id) {
                        it.copy(
                            installed = store.isInstalled(model.id, model.version),
                            downloadState = ModelDownloadState.SUCCEEDED,
                            bytesDownloaded = model.fileSizeBytes,
                            totalBytes = model.fileSizeBytes,
                            errorMessage = null,
                        )
                    }
                    removeObserver(requestId)
                }

                WorkInfo.State.FAILED -> {
                    updateCard(model.id) {
                        it.copy(
                            downloadState = ModelDownloadState.FAILED,
                            errorMessage = info.outputData.getString(ModelDownloadWorker.KEY_ERROR)
                                ?: "モデルのダウンロードに失敗しました",
                        )
                    }
                    removeObserver(requestId)
                }

                WorkInfo.State.CANCELLED -> {
                    updateCard(model.id) {
                        it.copy(
                            downloadState = ModelDownloadState.IDLE,
                            bytesDownloaded = 0L,
                        )
                    }
                    removeObserver(requestId)
                }
            }
        }
        workObservers[requestId] = liveData to observer
        liveData.observeForever(observer)
    }

    private fun removeObserver(requestId: UUID) {
        workObservers.remove(requestId)?.let { (liveData, observer) ->
            liveData.removeObserver(observer)
        }
    }

    private fun updateCard(modelId: String, transform: (ModelCardUiState) -> ModelCardUiState) {
        _state.value = _state.value.copy(
            cards = _state.value.cards.map { card ->
                if (card.model.id == modelId) transform(card) else card
            },
        )
    }

    override fun onCleared() {
        workObservers.values.forEach { (liveData, observer) ->
            liveData.removeObserver(observer)
        }
        workObservers.clear()
        super.onCleared()
    }

    private fun warningFor(capabilities: DeviceCapabilities?): String? = when (capabilities?.supportTier) {
        SupportTier.NOT_RECOMMENDED -> "この端末では処理が重くなる可能性があります"
        SupportTier.UNSUPPORTED -> "この端末はv1の最低要件（64bit / Vulkan 1.2）を満たしていません"
        else -> null
    }

    private fun hasProductionDownloadUrl(url: String): Boolean = runCatching {
        val host = URI(url).host?.lowercase() ?: return@runCatching false
        host != "example.invalid" && !host.endsWith(".invalid")
    }.getOrDefault(false)
}

class ModelManagerViewModelFactory(
    context: Context,
) : ViewModelProvider.Factory {
    private val appContext = context.applicationContext

    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        require(modelClass.isAssignableFrom(ModelManagerViewModel::class.java)) {
            "Unsupported ViewModel class: ${modelClass.name}"
        }

        var initialError: String? = null
        val models = runCatching {
            val json = appContext.assets.open("model-manifest.json").bufferedReader().use { it.readText() }
            ModelManifestParser().parse(json).models
        }.getOrElse { error ->
            initialError = "モデル一覧の読み込みに失敗しました: ${error.message ?: "unknown error"}"
            emptyList()
        }

        @Suppress("UNCHECKED_CAST")
        return ModelManagerViewModel(
            models = models,
            store = InstalledModelStore(appContext),
            deviceDetector = DeviceCapabilityDetector(appContext),
            downloadManager = ModelDownloadManager(appContext),
            workManager = WorkManager.getInstance(appContext),
            initialError = initialError,
        ) as T
    }
}
