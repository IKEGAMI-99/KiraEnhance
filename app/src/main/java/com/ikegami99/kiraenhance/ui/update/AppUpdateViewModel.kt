package com.ikegami99.kiraenhance.ui.update

import android.content.Context
import androidx.lifecycle.LiveData
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.ikegami99.kiraenhance.BuildConfig
import com.ikegami99.kiraenhance.update.AndroidAppUpdateInstallPlatform
import com.ikegami99.kiraenhance.update.AppUpdateCheckResult
import com.ikegami99.kiraenhance.update.AppUpdateDownloadManager
import com.ikegami99.kiraenhance.update.AppUpdateDownloadWorker
import com.ikegami99.kiraenhance.update.AppUpdateInfo
import com.ikegami99.kiraenhance.update.AppUpdateInstallStep
import com.ikegami99.kiraenhance.update.AppUpdateInstaller
import com.ikegami99.kiraenhance.update.AppUpdateChecker
import com.ikegami99.kiraenhance.update.HttpUpdateManifestSource
import java.io.File
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class AppUpdateUiState(
    val currentVersionName: String,
    val stage: AppUpdateStage = AppUpdateStage.IDLE,
    val latestVersionName: String? = null,
    val updateInfo: AppUpdateInfo? = null,
    val bytesDownloaded: Long = 0L,
    val totalBytes: Long = 0L,
    val apkPath: String? = null,
    val statusMessage: String? = null,
    val errorMessage: String? = null,
)

class AppUpdateViewModel(
    private val checker: AppUpdateChecker,
    private val downloadManager: AppUpdateDownloadManager,
    private val workManager: WorkManager,
    private val installer: AppUpdateInstaller,
    private val currentVersionCode: Int,
    currentVersionName: String,
) : ViewModel() {
    private val _state = MutableStateFlow(
        AppUpdateUiState(currentVersionName = currentVersionName),
    )
    val state: StateFlow<AppUpdateUiState> = _state.asStateFlow()

    private var activeRequestId: UUID? = null
    private var workObserver: Pair<LiveData<WorkInfo?>, Observer<WorkInfo?>>? = null

    fun checkForUpdates() {
        if (_state.value.stage in setOf(AppUpdateStage.CHECKING, AppUpdateStage.DOWNLOADING)) return

        _state.value = _state.value.copy(
            stage = AppUpdateStage.CHECKING,
            latestVersionName = null,
            updateInfo = null,
            bytesDownloaded = 0L,
            totalBytes = 0L,
            apkPath = null,
            statusMessage = null,
            errorMessage = null,
        )

        viewModelScope.launch(Dispatchers.IO) {
            runCatching { checker.check(currentVersionCode) }
                .onSuccess(::applyCheckResult)
                .onFailure { error ->
                    _state.value = _state.value.copy(
                        stage = AppUpdateStage.ERROR,
                        errorMessage = error.message ?: "更新確認に失敗しました",
                    )
                }
        }
    }

    fun download() {
        val info = _state.value.updateInfo ?: return
        if (_state.value.stage == AppUpdateStage.DOWNLOADING) return

        val requestId = downloadManager.enqueue(info)
        removeWorkObserver()
        activeRequestId = requestId
        _state.value = _state.value.copy(
            stage = AppUpdateStage.DOWNLOADING,
            bytesDownloaded = 0L,
            totalBytes = 0L,
            apkPath = null,
            statusMessage = "ダウンロード開始を待っています",
            errorMessage = null,
        )
        observeDownload(requestId)
    }

    fun install() {
        val apkPath = _state.value.apkPath ?: return
        val apkFile = File(apkPath)

        runCatching { installer.begin(apkFile) }
            .onSuccess { step ->
                _state.value = _state.value.copy(
                    statusMessage = when (step) {
                        AppUpdateInstallStep.REQUEST_UNKNOWN_SOURCES ->
                            "KiraEnhanceからのアプリインストールを許可して戻ったあと、もう一度「インストール」を押してください"

                        AppUpdateInstallStep.OPEN_PACKAGE_INSTALLER ->
                            "Androidのインストール画面を開きました"
                    },
                    errorMessage = null,
                )
            }
            .onFailure { error ->
                _state.value = _state.value.copy(
                    errorMessage = error.message ?: "インストーラを開けませんでした",
                )
            }
    }

    private fun applyCheckResult(result: AppUpdateCheckResult) {
        _state.value = when (result) {
            is AppUpdateCheckResult.Available -> _state.value.copy(
                stage = AppUpdateStage.AVAILABLE,
                latestVersionName = result.info.versionName,
                updateInfo = result.info,
                statusMessage = "新しいバージョンがあります",
                errorMessage = null,
            )

            is AppUpdateCheckResult.UpToDate -> _state.value.copy(
                stage = AppUpdateStage.UP_TO_DATE,
                latestVersionName = result.latestVersionName,
                updateInfo = null,
                statusMessage = "最新バージョンです",
                errorMessage = null,
            )
        }
    }

    private fun observeDownload(requestId: UUID) {
        val liveData = workManager.getWorkInfoByIdLiveData(requestId)
        val observer = Observer<WorkInfo?> { info ->
            if (info == null || activeRequestId != requestId) return@Observer

            val downloaded = when (info.state) {
                WorkInfo.State.SUCCEEDED -> info.outputData.getLong(
                    AppUpdateDownloadWorker.KEY_BYTES_DOWNLOADED,
                    0L,
                )

                else -> info.progress.getLong(AppUpdateDownloadWorker.KEY_BYTES_DOWNLOADED, 0L)
            }
            val total = when (info.state) {
                WorkInfo.State.SUCCEEDED -> info.outputData.getLong(
                    AppUpdateDownloadWorker.KEY_TOTAL_BYTES,
                    downloaded,
                )

                else -> info.progress.getLong(AppUpdateDownloadWorker.KEY_TOTAL_BYTES, 0L)
            }

            when (info.state) {
                WorkInfo.State.ENQUEUED -> updateDownloadState(
                    stage = AppUpdateStage.DOWNLOADING,
                    downloaded = downloaded,
                    total = total,
                    status = "ダウンロード開始を待っています",
                )

                WorkInfo.State.BLOCKED -> updateDownloadState(
                    stage = AppUpdateStage.DOWNLOADING,
                    downloaded = downloaded,
                    total = total,
                    status = "ダウンロード条件が整うのを待っています",
                )

                WorkInfo.State.RUNNING -> updateDownloadState(
                    stage = AppUpdateStage.DOWNLOADING,
                    downloaded = downloaded,
                    total = total,
                    status = null,
                )

                WorkInfo.State.SUCCEEDED -> {
                    val apkPath = info.outputData.getString(AppUpdateDownloadWorker.KEY_APK_PATH)
                    if (apkPath.isNullOrBlank()) {
                        _state.value = _state.value.copy(
                            stage = AppUpdateStage.ERROR,
                            errorMessage = "検証済みAPKの保存先を取得できませんでした",
                        )
                    } else {
                        _state.value = _state.value.copy(
                            stage = AppUpdateStage.READY_TO_INSTALL,
                            bytesDownloaded = downloaded,
                            totalBytes = total,
                            apkPath = apkPath,
                            statusMessage = "APKのSHA-256検証が完了しました",
                            errorMessage = null,
                        )
                    }
                    finishObservation(requestId)
                }

                WorkInfo.State.FAILED -> {
                    _state.value = _state.value.copy(
                        stage = AppUpdateStage.ERROR,
                        errorMessage = info.outputData.getString(AppUpdateDownloadWorker.KEY_ERROR)
                            ?: "更新APKのダウンロードに失敗しました",
                    )
                    finishObservation(requestId)
                }

                WorkInfo.State.CANCELLED -> {
                    _state.value = _state.value.copy(
                        stage = AppUpdateStage.AVAILABLE,
                        bytesDownloaded = 0L,
                        totalBytes = 0L,
                        statusMessage = "ダウンロードを中止しました",
                    )
                    finishObservation(requestId)
                }
            }
        }
        workObserver = liveData to observer
        liveData.observeForever(observer)
    }

    private fun updateDownloadState(
        stage: AppUpdateStage,
        downloaded: Long,
        total: Long,
        status: String?,
    ) {
        _state.value = _state.value.copy(
            stage = stage,
            bytesDownloaded = downloaded,
            totalBytes = total,
            statusMessage = status,
            errorMessage = null,
        )
    }

    private fun finishObservation(requestId: UUID) {
        if (activeRequestId == requestId) activeRequestId = null
        removeWorkObserver()
    }

    private fun removeWorkObserver() {
        workObserver?.let { (liveData, observer) -> liveData.removeObserver(observer) }
        workObserver = null
    }

    override fun onCleared() {
        removeWorkObserver()
        activeRequestId = null
        super.onCleared()
    }
}

class AppUpdateViewModelFactory(
    context: Context,
) : ViewModelProvider.Factory {
    private val appContext = context.applicationContext

    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        require(modelClass.isAssignableFrom(AppUpdateViewModel::class.java)) {
            "Unsupported ViewModel class: ${modelClass.name}"
        }

        @Suppress("UNCHECKED_CAST")
        return AppUpdateViewModel(
            checker = AppUpdateChecker(HttpUpdateManifestSource()),
            downloadManager = AppUpdateDownloadManager(appContext),
            workManager = WorkManager.getInstance(appContext),
            installer = AppUpdateInstaller(AndroidAppUpdateInstallPlatform(appContext)),
            currentVersionCode = BuildConfig.VERSION_CODE,
            currentVersionName = BuildConfig.VERSION_NAME,
        ) as T
    }
}
