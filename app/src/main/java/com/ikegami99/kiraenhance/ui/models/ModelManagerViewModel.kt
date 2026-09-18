package com.ikegami99.kiraenhance.ui.models

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.LiveData
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.ikegami99.kiraenhance.device.DeviceCapabilities
import com.ikegami99.kiraenhance.device.DeviceCapabilityDetector
import com.ikegami99.kiraenhance.device.SupportTier
import com.ikegami99.kiraenhance.diagnostics.AppDiagnosticLogger
import com.ikegami99.kiraenhance.download.ModelDownloadManager
import com.ikegami99.kiraenhance.download.ModelDownloadWorker
import com.ikegami99.kiraenhance.model.InstalledModelStore
import com.ikegami99.kiraenhance.model.ModelDescriptor
import com.ikegami99.kiraenhance.model.ModelManifestParser
import java.net.URI
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

enum class ModelDownloadState {
    IDLE,
    QUEUED,
    RETRY_WAIT,
    BLOCKED,
    RUNNING,
    SUCCEEDED,
    FAILED,
}

internal data class ModelDownloadRestartDecision(
    val replaceExisting: Boolean,
    val deleteLocalFiles: Boolean,
)

internal object ModelDownloadUiReducer {
    fun stateFor(
        workState: WorkInfo.State,
        runAttemptCount: Int = 0,
    ): ModelDownloadState = when (workState) {
        WorkInfo.State.ENQUEUED -> if (runAttemptCount > 0) {
            ModelDownloadState.RETRY_WAIT
        } else {
            ModelDownloadState.QUEUED
        }

        WorkInfo.State.BLOCKED -> ModelDownloadState.BLOCKED
        WorkInfo.State.RUNNING -> ModelDownloadState.RUNNING
        WorkInfo.State.SUCCEEDED -> ModelDownloadState.SUCCEEDED
        WorkInfo.State.FAILED -> ModelDownloadState.FAILED
        WorkInfo.State.CANCELLED -> ModelDownloadState.IDLE
    }

    fun waitingMessage(downloadState: ModelDownloadState, wifiOnly: Boolean): String? = when (downloadState) {
        ModelDownloadState.QUEUED -> if (wifiOnly) {
            "Wi‑Fiのみ設定のため、非従量制ネットワークを待っています"
        } else {
            "ネットワーク接続または実行開始を待っています"
        }

        ModelDownloadState.RETRY_WAIT -> "ダウンロード再試行を待っています"
        ModelDownloadState.BLOCKED -> "前提となる処理の完了を待っています"
        else -> null
    }

    fun restartDecision(
        previousWifiOnly: Boolean,
        newWifiOnly: Boolean,
        state: ModelDownloadState,
    ): ModelDownloadRestartDecision? =
        if (
            previousWifiOnly &&
            !newWifiOnly &&
            state in setOf(
                ModelDownloadState.QUEUED,
                ModelDownloadState.RETRY_WAIT,
                ModelDownloadState.BLOCKED,
            )
        ) {
            ModelDownloadRestartDecision(
                replaceExisting = true,
                deleteLocalFiles = false,
            )
        } else {
            null
        }
}

data class ModelCardUiState(
    val model: ModelDescriptor,
    val installed: Boolean,
    val downloadState: ModelDownloadState = ModelDownloadState.IDLE,
    val bytesDownloaded: Long = 0L,
    val totalBytes: Long = model.totalFileSizeBytes,
    val warningMessage: String? = null,
    val errorMessage: String? = null,
    val downloadAvailable: Boolean = true,
    val validationImporting: Boolean = false,
)

data class ModelManagerUiState(
    val cards: List<ModelCardUiState> = emptyList(),
    val deviceCapabilities: DeviceCapabilities? = null,
    val wifiOnly: Boolean = false,
    val errorMessage: String? = null,
)

class ModelManagerViewModel(
    private val models: List<ModelDescriptor>,
    private val store: InstalledModelStore,
    private val deviceDetector: DeviceCapabilityDetector,
    private val downloadManager: ModelDownloadManager,
    private val workManager: WorkManager,
    private val contentResolver: ContentResolver,
    private val logger: AppDiagnosticLogger,
    initialError: String? = null,
) : ViewModel() {
    private val _state = MutableStateFlow(ModelManagerUiState(errorMessage = initialError))
    val state: StateFlow<ModelManagerUiState> = _state.asStateFlow()

    private val workObservers = mutableMapOf<UUID, Pair<LiveData<WorkInfo?>, Observer<WorkInfo?>>>()
    private val activeRequestIds = mutableMapOf<String, UUID>()

    init {
        refreshCards()
    }

    fun setWifiOnly(enabled: Boolean) {
        val previous = _state.value
        _state.value = previous.copy(wifiOnly = enabled)
        logger.log("ModelManager", "wifiOnly changed from=${previous.wifiOnly} to=$enabled")

        previous.cards.forEach { card ->
            val decision = ModelDownloadUiReducer.restartDecision(
                previousWifiOnly = previous.wifiOnly,
                newWifiOnly = enabled,
                state = card.downloadState,
            ) ?: return@forEach

            check(!decision.deleteLocalFiles)
            enqueueDownload(card.model, replaceExisting = decision.replaceExisting)
            updateCard(card.model.id) {
                it.copy(
                    downloadState = ModelDownloadState.QUEUED,
                    errorMessage = null,
                )
            }
        }
    }

    fun download(modelId: String) {
        val card = _state.value.cards.firstOrNull { it.model.id == modelId } ?: return
        logger.log(
            "ModelManager",
            "download tapped model=$modelId state=${card.downloadState} installed=${card.installed} " +
                "available=${card.downloadAvailable} wifiOnly=${_state.value.wifiOnly}",
        )
        if (!card.downloadAvailable) {
            updateCard(modelId) {
                it.copy(errorMessage = "このモデルは配布URLの準備中です")
            }
            return
        }
        if (card.downloadState in setOf(
                ModelDownloadState.RUNNING,
                ModelDownloadState.QUEUED,
                ModelDownloadState.RETRY_WAIT,
                ModelDownloadState.BLOCKED,
            )
        ) {
            logger.log("ModelManager", "download ignored because work is busy model=$modelId")
            return
        }

        val reinstalling = card.installed
        if (reinstalling && !removeLocalModelFiles(card)) return

        enqueueDownload(
            model = card.model,
            replaceExisting = reinstalling,
        )
        updateCard(modelId) {
            it.copy(
                installed = if (reinstalling) false else it.installed,
                downloadState = ModelDownloadState.QUEUED,
                bytesDownloaded = 0L,
                totalBytes = card.model.totalFileSizeBytes,
                errorMessage = null,
            )
        }
    }

    fun delete(modelId: String) {
        val card = _state.value.cards.firstOrNull { it.model.id == modelId } ?: return
        logger.log("ModelManager", "delete model=$modelId")
        activeRequestIds.remove(modelId)?.let(::removeObserver)
        workManager.cancelUniqueWork(ModelDownloadManager.workName(card.model))
        if (!removeLocalModelFiles(card)) return

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
        download(modelId)
    }

    fun importValidationArtifacts(modelId: String, uris: List<Uri>) {
        val card = _state.value.cards.firstOrNull { it.model.id == modelId } ?: return
        if (card.downloadAvailable) {
            updateCard(modelId) {
                it.copy(errorMessage = "配布済みモデルではローカル検証インポートを使用できません")
            }
            return
        }
        if (uris.isEmpty()) {
            logger.log("ModelImport", "cancelled model=$modelId")
            return
        }

        updateCard(modelId) {
            it.copy(
                validationImporting = true,
                errorMessage = null,
            )
        }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                importValidationArtifacts(card.model, uris)
            }.fold(
                onSuccess = { bytes ->
                    logger.log(
                        "ModelImport",
                        "success model=$modelId files=${card.model.artifacts.size} bytes=$bytes",
                    )
                    updateCard(modelId) {
                        it.copy(
                            installed = store.isInstalled(card.model),
                            downloadState = ModelDownloadState.SUCCEEDED,
                            validationImporting = false,
                            errorMessage = null,
                        )
                    }
                },
                onFailure = { error ->
                    logger.log(
                        "ModelImport",
                        "failed model=$modelId type=${error.javaClass.simpleName} " +
                            "message=${error.message ?: "no-message"}",
                    )
                    updateCard(modelId) {
                        it.copy(
                            installed = store.isInstalled(card.model),
                            downloadState = ModelDownloadState.FAILED,
                            validationImporting = false,
                            errorMessage = error.message ?: "検証用モデルの読み込みに失敗しました",
                        )
                    }
                },
            )
        }
    }

    private fun importValidationArtifacts(model: ModelDescriptor, uris: List<Uri>): Long {
        val requiredNames = model.artifacts.map { it.fileName }
        val requiredSet = requiredNames.toSet()
        val selected = linkedMapOf<String, Uri>()

        uris.forEach { uri ->
            val name = displayName(uri)
                ?: error("選択したファイル名を取得できませんでした")
            if (name in requiredSet) {
                check(selected.put(name, uri) == null) {
                    "同じファイルが複数選択されています: $name"
                }
            }
        }

        val missing = requiredNames.filterNot(selected::containsKey)
        check(missing.isEmpty()) {
            "必要なファイルが不足しています: ${missing.joinToString()}"
        }

        check(store.deleteModel(model)) {
            "既存の検証用モデルを削除できませんでした"
        }

        var copiedBytes = 0L
        try {
            requiredNames.forEach { fileName ->
                val sourceUri = selected.getValue(fileName)
                val destination = store.artifactFile(model.id, model.version, fileName)
                destination.parentFile?.mkdirs()
                val copied = contentResolver.openInputStream(sourceUri)?.use { input ->
                    destination.outputStream().use { output ->
                        input.copyTo(output)
                    }
                } ?: error("ファイルを開けませんでした: $fileName")
                check(copied > 0L && destination.isFile && destination.length() > 0L) {
                    "空のファイルは読み込めません: $fileName"
                }
                copiedBytes += copied
            }

            check(store.markValidationInstalled(model)) {
                "検証用モデルのインストール情報を保存できませんでした"
            }
            check(store.isInstalled(model)) {
                "検証用モデルの保存後チェックに失敗しました"
            }
            return copiedBytes
        } catch (error: Throwable) {
            store.deleteModel(model)
            throw error
        }
    }

    private fun displayName(uri: Uri): String? {
        val fromProvider = contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0 && cursor.moveToFirst()) cursor.getString(index) else null
        }
        return fromProvider
            ?.takeIf { it.isNotBlank() }
            ?: uri.lastPathSegment
                ?.substringAfterLast('/')
                ?.takeIf { it.isNotBlank() }
    }

    private fun enqueueDownload(model: ModelDescriptor, replaceExisting: Boolean): UUID {
        val requestId = downloadManager.enqueue(
            model = model,
            wifiOnly = _state.value.wifiOnly,
            replaceExisting = replaceExisting,
        )
        activeRequestIds.put(model.id, requestId)?.let(::removeObserver)
        observeDownload(model, requestId)
        return requestId
    }

    private fun removeLocalModelFiles(card: ModelCardUiState): Boolean = runCatching {
        check(store.deleteModel(card.model)) { "モデルファイルを削除できませんでした" }
    }.fold(
        onSuccess = { true },
        onFailure = { error ->
            logger.log(
                "ModelManager",
                "delete failed model=${card.model.id} type=${error.javaClass.simpleName} " +
                    "message=${error.message ?: "no-message"}",
            )
            updateCard(card.model.id) {
                it.copy(errorMessage = error.message ?: "モデルの削除に失敗しました")
            }
            false
        },
    )

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
                    installed = store.isInstalled(model),
                    warningMessage = warning,
                    downloadAvailable = model.artifacts.all { artifact ->
                        hasProductionDownloadUrl(artifact.downloadUrl)
                    },
                )
            },
        )
    }

    private fun observeDownload(model: ModelDescriptor, requestId: UUID) {
        val liveData = workManager.getWorkInfoByIdLiveData(requestId)
        val observer = Observer<WorkInfo?> { info ->
            if (info == null || activeRequestIds[model.id] != requestId) return@Observer

            val downloaded = info.progress.getLong(ModelDownloadWorker.KEY_BYTES_DOWNLOADED, 0L)
            val total = info.progress.getLong(ModelDownloadWorker.KEY_TOTAL_BYTES, model.totalFileSizeBytes)
                .takeIf { it > 0L }
                ?: model.totalFileSizeBytes
            val uiState = ModelDownloadUiReducer.stateFor(info.state, info.runAttemptCount)

            logger.log(
                "WorkManager",
                "model=${model.id} request=$requestId state=${info.state} uiState=$uiState " +
                    "attempt=${info.runAttemptCount} bytes=$downloaded/$total",
            )

            when (info.state) {
                WorkInfo.State.ENQUEUED -> updateCard(model.id) {
                    it.copy(
                        downloadState = uiState,
                        bytesDownloaded = downloaded,
                        totalBytes = total,
                    )
                }

                WorkInfo.State.BLOCKED -> updateCard(model.id) {
                    it.copy(
                        downloadState = ModelDownloadState.BLOCKED,
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
                            installed = store.isInstalled(model),
                            downloadState = ModelDownloadState.SUCCEEDED,
                            bytesDownloaded = model.totalFileSizeBytes,
                            totalBytes = model.totalFileSizeBytes,
                            errorMessage = null,
                        )
                    }
                    finishObservation(model.id, requestId)
                }

                WorkInfo.State.FAILED -> {
                    val error = info.outputData.getString(ModelDownloadWorker.KEY_ERROR)
                        ?: "モデルのダウンロードに失敗しました"
                    logger.log(
                        "WorkManager",
                        "failed model=${model.id} request=$requestId attempt=${info.runAttemptCount} error=$error",
                    )
                    updateCard(model.id) {
                        it.copy(
                            installed = store.isInstalled(model),
                            downloadState = ModelDownloadState.FAILED,
                            errorMessage = error,
                        )
                    }
                    finishObservation(model.id, requestId)
                }

                WorkInfo.State.CANCELLED -> {
                    logger.log("WorkManager", "cancelled model=${model.id} request=$requestId")
                    updateCard(model.id) {
                        it.copy(
                            installed = store.isInstalled(model),
                            downloadState = ModelDownloadState.IDLE,
                            bytesDownloaded = 0L,
                        )
                    }
                    finishObservation(model.id, requestId)
                }
            }
        }
        workObservers[requestId] = liveData to observer
        liveData.observeForever(observer)
    }

    private fun finishObservation(modelId: String, requestId: UUID) {
        activeRequestIds.remove(modelId, requestId)
        removeObserver(requestId)
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
        activeRequestIds.clear()
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
            contentResolver = appContext.contentResolver,
            logger = AppDiagnosticLogger.get(appContext),
            initialError = initialError,
        ) as T
    }
}
