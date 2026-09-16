# Model Download Wait Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model downloads start on ordinary connected networks by default, explain WorkManager waiting states clearly, and safely restart queued work when Wi-Fi-only is disabled.

**Architecture:** Keep WorkManager and the existing `ModelDownloadWorker`, including resume/checksum/HTTP 416 behavior. Add a pure reducer for WorkInfo waiting states, make `wifiOnly` default false, encode restart behavior as a tested decision that explicitly preserves partial files, and guard observers by active request ID so replaced work cannot overwrite newer state.

**Tech Stack:** Kotlin, Android WorkManager 2.11.2, Compose Material 3, JUnit 4, Android Compose UI tests.

**Spec:** `docs/superpowers/specs/2026-09-16-model-download-and-app-update-design.md`

## Global Constraints

- Android `minSdk = 28`, `targetSdk = 36`.
- Default model download network policy is `NetworkType.CONNECTED`.
- Wi-Fi-only mode maps to `NetworkType.UNMETERED`.
- Existing partial-file resume, SHA-256 verification, and HTTP 416 restart-from-zero behavior remain unchanged.
- Switching Wi-Fi-only off while queued/blocked replaces the unique work without deleting partial files.
- Stale callbacks from replaced WorkManager requests do not mutate current UI state.

---

### Task 1: Add explicit waiting states and messages

**Files:**
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerViewModel.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/ui/models/ModelDownloadUiReducerTest.kt`

**Interfaces:**
- Produces: `ModelDownloadState.BLOCKED`
- Produces: `ModelDownloadUiReducer.stateFor(workState: WorkInfo.State): ModelDownloadState`
- Produces: `ModelDownloadUiReducer.waitingMessage(downloadState: ModelDownloadState, wifiOnly: Boolean): String?`

- [ ] **Step 1: Write the failing reducer tests**

```kotlin
package com.ikegami99.kiraenhance.ui.models

import androidx.work.WorkInfo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ModelDownloadUiReducerTest {
    @Test
    fun queuedAndBlockedRemainDistinct() {
        assertEquals(ModelDownloadState.QUEUED, ModelDownloadUiReducer.stateFor(WorkInfo.State.ENQUEUED))
        assertEquals(ModelDownloadState.BLOCKED, ModelDownloadUiReducer.stateFor(WorkInfo.State.BLOCKED))
    }

    @Test
    fun waitingMessagesExplainWhyWorkHasNotStarted() {
        assertEquals(
            "Wi‑Fiのみ設定のため、非従量制ネットワークを待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.QUEUED, wifiOnly = true),
        )
        assertEquals(
            "ネットワーク接続または実行開始を待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.QUEUED, wifiOnly = false),
        )
        assertEquals(
            "前提となる処理の完了を待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.BLOCKED, wifiOnly = false),
        )
        assertNull(ModelDownloadUiReducer.waitingMessage(ModelDownloadState.RUNNING, wifiOnly = false))
    }
}
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.ui.models.ModelDownloadUiReducerTest' --stacktrace
```

Expected: compilation failure because `ModelDownloadUiReducer` and `ModelDownloadState.BLOCKED` do not exist.

- [ ] **Step 3: Implement the reducer and distinct WorkInfo mapping**

Add `BLOCKED` to `ModelDownloadState`, then add:

```kotlin
internal object ModelDownloadUiReducer {
    fun stateFor(workState: WorkInfo.State): ModelDownloadState = when (workState) {
        WorkInfo.State.ENQUEUED -> ModelDownloadState.QUEUED
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
        ModelDownloadState.BLOCKED -> "前提となる処理の完了を待っています"
        else -> null
    }
}
```

Replace the shared `ENQUEUED/BLOCKED` observer branch with:

```kotlin
WorkInfo.State.ENQUEUED -> updateCard(model.id) {
    it.copy(
        downloadState = ModelDownloadState.QUEUED,
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
```

- [ ] **Step 4: Run the focused test and verify GREEN**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.ui.models.ModelDownloadUiReducerTest' --stacktrace
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerViewModel.kt app/src/test/java/com/ikegami99/kiraenhance/ui/models/ModelDownloadUiReducerTest.kt
git commit -m "fix: expose model download wait states"
```

---

### Task 2: Default downloads to CONNECTED and restart waiting work without deleting partials

**Files:**
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerViewModel.kt`
- Modify: `app/src/test/java/com/ikegami99/kiraenhance/download/ModelDownloadManagerTest.kt`
- Modify: `app/src/test/java/com/ikegami99/kiraenhance/ui/models/ModelDownloadUiReducerTest.kt`

**Interfaces:**
- Consumes: `ModelDownloadManager.enqueue(model: ModelDescriptor, wifiOnly: Boolean, replaceExisting: Boolean): UUID`
- Produces: `ModelManagerUiState.wifiOnly = false`
- Produces: `data class ModelDownloadRestartDecision(val replaceExisting: Boolean, val deleteLocalFiles: Boolean)`
- Produces: `ModelDownloadUiReducer.restartDecision(previousWifiOnly: Boolean, newWifiOnly: Boolean, state: ModelDownloadState): ModelDownloadRestartDecision?`
- Produces: `activeRequestIds: MutableMap<String, UUID>` in `ModelManagerViewModel`

- [ ] **Step 1: Add failing restart-policy tests**

Append to `ModelDownloadUiReducerTest.kt`:

```kotlin
@Test
fun disablingWifiOnlyReplacesWaitingWorkAndPreservesPartialFiles() {
    val queued = ModelDownloadUiReducer.restartDecision(
        previousWifiOnly = true,
        newWifiOnly = false,
        state = ModelDownloadState.QUEUED,
    )
    assertEquals(ModelDownloadRestartDecision(replaceExisting = true, deleteLocalFiles = false), queued)

    val blocked = ModelDownloadUiReducer.restartDecision(
        previousWifiOnly = true,
        newWifiOnly = false,
        state = ModelDownloadState.BLOCKED,
    )
    assertEquals(ModelDownloadRestartDecision(replaceExisting = true, deleteLocalFiles = false), blocked)

    assertNull(
        ModelDownloadUiReducer.restartDecision(
            previousWifiOnly = true,
            newWifiOnly = false,
            state = ModelDownloadState.RUNNING,
        ),
    )
}

@Test
fun modelManagerDefaultsWifiOnlyOff() {
    assertEquals(false, ModelManagerUiState().wifiOnly)
}
```

`ModelDownloadManagerTest` already proves `wifiOnly = false` maps to `NetworkType.CONNECTED`; retain that test as the network-policy regression.

- [ ] **Step 2: Run tests and verify RED**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.ui.models.ModelDownloadUiReducerTest' --tests 'com.ikegami99.kiraenhance.download.ModelDownloadManagerTest' --stacktrace
```

Expected: compilation failure because `ModelDownloadRestartDecision` and `restartDecision` do not exist, and the default is still true.

- [ ] **Step 3: Implement the tested restart decision**

Add:

```kotlin
data class ModelDownloadRestartDecision(
    val replaceExisting: Boolean,
    val deleteLocalFiles: Boolean,
)
```

Add to `ModelDownloadUiReducer`:

```kotlin
fun restartDecision(
    previousWifiOnly: Boolean,
    newWifiOnly: Boolean,
    state: ModelDownloadState,
): ModelDownloadRestartDecision? =
    if (
        previousWifiOnly &&
        !newWifiOnly &&
        state in setOf(ModelDownloadState.QUEUED, ModelDownloadState.BLOCKED)
    ) {
        ModelDownloadRestartDecision(
            replaceExisting = true,
            deleteLocalFiles = false,
        )
    } else {
        null
    }
```

Change `ModelManagerUiState` to:

```kotlin
val wifiOnly: Boolean = false,
```

- [ ] **Step 4: Track the active request and restart without file deletion**

Add:

```kotlin
private val activeRequestIds = mutableMapOf<String, UUID>()
```

Add this helper:

```kotlin
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
```

Implement `setWifiOnly` as:

```kotlin
fun setWifiOnly(enabled: Boolean) {
    val previous = _state.value
    _state.value = previous.copy(wifiOnly = enabled)

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
```

Do not call `removeLocalModelFiles` from this network-setting restart path. Keep `removeLocalModelFiles` only for delete/reinstall behavior.

At the beginning of the `Observer<WorkInfo?>` callback add:

```kotlin
if (activeRequestIds[model.id] != requestId) return@Observer
```

On `SUCCEEDED`, `FAILED`, and `CANCELLED`, use:

```kotlin
activeRequestIds.remove(model.id, requestId)
removeObserver(requestId)
```

Change the ordinary `download()` path to call `enqueueDownload(card.model, replaceExisting = reinstalling)` instead of calling `downloadManager.enqueue` directly.

- [ ] **Step 5: Run full unit tests**

```bash
gradle :app:testDebugUnitTest --stacktrace
```

Expected: PASS, including `ModelDownloadResumeTest` for HTTP 416.

- [ ] **Step 6: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerViewModel.kt app/src/test/java/com/ikegami99/kiraenhance/download/ModelDownloadManagerTest.kt app/src/test/java/com/ikegami99/kiraenhance/ui/models/ModelDownloadUiReducerTest.kt
git commit -m "fix: restart waiting model downloads on connected networks"
```

---

### Task 3: Render the waiting reason and verify the branch build

**Files:**
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreen.kt`
- Modify: `app/src/androidTest/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreenTest.kt`

**Interfaces:**
- Consumes: `ModelDownloadUiReducer.waitingMessage(downloadState: ModelDownloadState, wifiOnly: Boolean): String?`
- Produces: `ModelCard(card: ModelCardUiState, wifiOnly: Boolean, onDownload: () -> Unit, onDelete: () -> Unit, onOpenLicense: () -> Unit)`

- [ ] **Step 1: Add failing Compose UI assertions**

Add one test rendering a queued card with `wifiOnly = true` and assert:

```kotlin
composeRule.onNodeWithText(
    "Wi‑Fiのみ設定のため、非従量制ネットワークを待っています",
).assertIsDisplayed()
```

Add one test rendering a blocked card and assert:

```kotlin
composeRule.onNodeWithText("前提となる処理の完了を待っています").assertIsDisplayed()
```

- [ ] **Step 2: Compile androidTest and verify RED**

```bash
gradle :app:assembleDebugAndroidTest --stacktrace
```

Expected: the new UI test does not pass until `BLOCKED` and `wifiOnly` are rendered distinctly.

- [ ] **Step 3: Pass Wi-Fi policy into each card and render the reducer message**

Change the call to:

```kotlin
ModelCard(
    card = card,
    wifiOnly = state.wifiOnly,
    onDownload = { onDownload(card.model.id) },
    onDelete = { onDelete(card.model.id) },
    onOpenLicense = { onOpenLicense(card.model.licenseUrl) },
)
```

Change the signature to:

```kotlin
private fun ModelCard(
    card: ModelCardUiState,
    wifiOnly: Boolean,
    onDownload: () -> Unit,
    onDelete: () -> Unit,
    onOpenLicense: () -> Unit,
)
```

Replace the current queued/running block with:

```kotlin
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
    LinearProgressIndicator(
        progress = { progress },
        modifier = Modifier.fillMaxWidth(),
    )
    val waitingMessage = ModelDownloadUiReducer.waitingMessage(card.downloadState, wifiOnly)
    Text(
        text = waitingMessage
            ?: "${(progress * 100).toInt()}% • ${formatBytes(card.bytesDownloaded)} / ${formatBytes(card.totalBytes)}",
        style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}
```

- [ ] **Step 4: Run CI-equivalent verification**

```bash
gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace
```

Expected: all three tasks complete successfully.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreen.kt app/src/androidTest/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreenTest.kt
git commit -m "fix: explain model download waiting state"
```

---

### Task 4: Real-device acceptance

**Files:**
- No source files.

**Interfaces:**
- Validates real Android network classification and partial-resume behavior.

- [ ] **Step 1: Install the branch CI debug APK on the target phone**

Use the APK artifact built from the final commit of Tasks 1-3.

- [ ] **Step 2: Verify default connected-network behavior**

Open Model Manager and confirm `Wi‑Fiのみでモデルをダウンロード` is OFF by default. Tap UltraSharp `ダウンロード` and verify state reaches RUNNING instead of remaining indefinitely at QUEUED.

- [ ] **Step 3: Verify Wi-Fi-only restart**

Turn Wi-Fi-only ON, start UltraSharp on a network Android does not classify as unmetered, confirm the explanatory waiting message, then turn Wi-Fi-only OFF and confirm the same work is replaced and proceeds without losing already-downloaded partial bytes.

- [ ] **Step 4: Verify integrity**

Confirm UltraSharp becomes installed only after both model artifacts match their manifest file sizes and SHA-256 values.
