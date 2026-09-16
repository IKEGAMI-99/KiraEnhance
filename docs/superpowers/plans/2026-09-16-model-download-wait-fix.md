# Model Download Wait Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model downloads start on ordinary connected networks by default, explain WorkManager waiting states clearly, and safely restart queued work when Wi-Fi-only is disabled.

**Architecture:** Keep WorkManager and the existing `ModelDownloadWorker`, including resume/checksum/HTTP 416 behavior. Add a small pure UI-state reducer for WorkInfo-state-to-message mapping, make `wifiOnly` default false, and guard observers by active request ID so replaced work cannot overwrite newer state.

**Tech Stack:** Kotlin, Android WorkManager 2.11.2, Compose Material 3, JUnit 4, Android Compose UI tests.

**Spec:** `docs/superpowers/specs/2026-09-16-model-download-and-app-update-design.md`

## Global Constraints

- Android `minSdk = 28`, `targetSdk = 36`.
- Default model download network policy is `NetworkType.CONNECTED`.
- Wi-Fi-only mode remains available and maps to `NetworkType.UNMETERED`.
- Existing partial-file resume, SHA-256 verification, and HTTP 416 restart-from-zero behavior must remain unchanged.
- Switching Wi-Fi-only off while queued/blocked must replace the unique work without deleting partial files.
- Stale callbacks from replaced WorkManager requests must not mutate current UI state.

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
    fun queuedWifiOnlyExplainsUnmeteredWait() {
        assertEquals(
            "Wi‑Fiのみ設定のため、非従量制ネットワークを待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.QUEUED, wifiOnly = true),
        )
    }

    @Test
    fun queuedConnectedExplainsGenericWait() {
        assertEquals(
            "ネットワーク接続または実行開始を待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.QUEUED, wifiOnly = false),
        )
    }

    @Test
    fun blockedExplainsPrerequisiteWait() {
        assertEquals(
            "前提となる処理の完了を待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.BLOCKED, wifiOnly = false),
        )
    }

    @Test
    fun runningHasNoWaitingMessage() {
        assertNull(ModelDownloadUiReducer.waitingMessage(ModelDownloadState.RUNNING, wifiOnly = false))
    }
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.ui.models.ModelDownloadUiReducerTest' --stacktrace
```

Expected: compilation failure because `ModelDownloadUiReducer` and `ModelDownloadState.BLOCKED` do not exist.

- [ ] **Step 3: Implement the minimal reducer and enum state**

Add `BLOCKED` to `ModelDownloadState`, then add this internal object in `ModelManagerViewModel.kt`:

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

Update the WorkInfo observer so `ENQUEUED` and `BLOCKED` are handled separately instead of sharing one branch.

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

### Task 2: Default downloads to CONNECTED and safely replace waiting work

**Files:**
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerViewModel.kt`
- Modify: `app/src/test/java/com/ikegami99/kiraenhance/download/ModelDownloadManagerTest.kt`
- Modify: `app/src/test/java/com/ikegami99/kiraenhance/ui/models/ModelDownloadUiReducerTest.kt`

**Interfaces:**
- Consumes: `ModelDownloadManager.enqueue(model, wifiOnly, replaceExisting)`
- Produces: `ModelManagerUiState.wifiOnly = false`
- Produces: `ModelDownloadUiReducer.shouldReplaceForNetworkChange(previousWifiOnly: Boolean, newWifiOnly: Boolean, state: ModelDownloadState): Boolean`
- Produces: `activeRequestIds: MutableMap<String, UUID>` in `ModelManagerViewModel`

- [ ] **Step 1: Add failing policy tests**

Append:

```kotlin
@Test
fun disablingWifiOnlyRestartsWaitingWork() {
    assertTrue(
        ModelDownloadUiReducer.shouldReplaceForNetworkChange(
            previousWifiOnly = true,
            newWifiOnly = false,
            state = ModelDownloadState.QUEUED,
        ),
    )
    assertTrue(
        ModelDownloadUiReducer.shouldReplaceForNetworkChange(
            previousWifiOnly = true,
            newWifiOnly = false,
            state = ModelDownloadState.BLOCKED,
        ),
    )
    assertFalse(
        ModelDownloadUiReducer.shouldReplaceForNetworkChange(
            previousWifiOnly = true,
            newWifiOnly = false,
            state = ModelDownloadState.RUNNING,
        ),
    )
}
```

Also add to `ModelDownloadManagerTest`:

```kotlin
@Test
fun defaultPolicyUsesConnectedNetwork() {
    assertEquals(NetworkType.CONNECTED, ModelDownloadManager.networkTypeFor(wifiOnly = false))
}
```

Import `assertTrue` and `assertFalse` where needed.

- [ ] **Step 2: Run tests and verify RED**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.ui.models.ModelDownloadUiReducerTest' --tests 'com.ikegami99.kiraenhance.download.ModelDownloadManagerTest' --stacktrace
```

Expected: `shouldReplaceForNetworkChange` unresolved.

- [ ] **Step 3: Implement restart policy and stale-observer guard**

Add:

```kotlin
fun shouldReplaceForNetworkChange(
    previousWifiOnly: Boolean,
    newWifiOnly: Boolean,
    state: ModelDownloadState,
): Boolean = previousWifiOnly && !newWifiOnly && state in setOf(
    ModelDownloadState.QUEUED,
    ModelDownloadState.BLOCKED,
)
```

Change `ModelManagerUiState` to:

```kotlin
val wifiOnly: Boolean = false,
```

Add to the ViewModel:

```kotlin
private val activeRequestIds = mutableMapOf<String, UUID>()
```

Refactor enqueueing into:

```kotlin
private fun enqueueDownload(model: ModelDescriptor, replaceExisting: Boolean) {
    val requestId = downloadManager.enqueue(
        model = model,
        wifiOnly = _state.value.wifiOnly,
        replaceExisting = replaceExisting,
    )
    activeRequestIds.put(model.id, requestId)?.let(::removeObserver)
    observeDownload(model, requestId)
}
```

In `setWifiOnly`, capture the previous state, update the flag, then restart only waiting models without deleting model or partial files:

```kotlin
fun setWifiOnly(enabled: Boolean) {
    val previous = _state.value
    _state.value = previous.copy(wifiOnly = enabled)

    previous.cards
        .filter { card ->
            ModelDownloadUiReducer.shouldReplaceForNetworkChange(
                previousWifiOnly = previous.wifiOnly,
                newWifiOnly = enabled,
                state = card.downloadState,
            )
        }
        .forEach { card ->
            enqueueDownload(card.model, replaceExisting = true)
            updateCard(card.model.id) {
                it.copy(downloadState = ModelDownloadState.QUEUED, errorMessage = null)
            }
        }
}
```

At the top of each WorkInfo observer callback, ignore stale request IDs:

```kotlin
if (activeRequestIds[model.id] != requestId) return@Observer
```

On terminal states, remove the active ID only when it still equals the callback request:

```kotlin
activeRequestIds.remove(model.id, requestId)
removeObserver(requestId)
```

Use `enqueueDownload(...)` from the ordinary `download()` path so request tracking is consistent.

- [ ] **Step 4: Run focused tests and full unit tests**

```bash
gradle :app:testDebugUnitTest --stacktrace
```

Expected: PASS, including the existing HTTP 416 regression test.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerViewModel.kt app/src/test/java/com/ikegami99/kiraenhance/download/ModelDownloadManagerTest.kt app/src/test/java/com/ikegami99/kiraenhance/ui/models/ModelDownloadUiReducerTest.kt
git commit -m "fix: restart model downloads when wifi-only is disabled"
```

---

### Task 3: Show useful waiting text in the model manager UI

**Files:**
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreen.kt`
- Modify: `app/src/androidTest/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreenTest.kt`

**Interfaces:**
- Consumes: `ModelDownloadUiReducer.waitingMessage(...)`
- Produces: `ModelCard(card, wifiOnly, ...)` rendering distinct queued/blocked text.

- [ ] **Step 1: Add failing Compose UI assertions**

Create card state fixtures for queued Wi-Fi-only and blocked states, then assert these exact texts are displayed:

```kotlin
composeRule.onNodeWithText(
    "Wi‑Fiのみ設定のため、非従量制ネットワークを待っています",
).assertIsDisplayed()
```

and:

```kotlin
composeRule.onNodeWithText("前提となる処理の完了を待っています").assertIsDisplayed()
```

- [ ] **Step 2: Compile androidTest and verify RED**

```bash
gradle :app:assembleDebugAndroidTest --stacktrace
```

Expected: test compilation or assertion fixture fails until the new `BLOCKED` rendering path is implemented.

- [ ] **Step 3: Pass `state.wifiOnly` into each model card and render the reducer message**

Change the call:

```kotlin
ModelCard(
    card = card,
    wifiOnly = state.wifiOnly,
    onDownload = { onDownload(card.model.id) },
    onDelete = { onDelete(card.model.id) },
    onOpenLicense = { onOpenLicense(card.model.licenseUrl) },
)
```

Change the signature:

```kotlin
private fun ModelCard(
    card: ModelCardUiState,
    wifiOnly: Boolean,
    onDownload: () -> Unit,
    onDelete: () -> Unit,
    onOpenLicense: () -> Unit,
)
```

Render progress for both waiting states and use the reducer text:

```kotlin
if (card.downloadState in setOf(ModelDownloadState.QUEUED, ModelDownloadState.BLOCKED, ModelDownloadState.RUNNING)) {
    val progress = if (card.totalBytes > 0L) {
        (card.bytesDownloaded.toFloat() / card.totalBytes.toFloat()).coerceIn(0f, 1f)
    } else {
        0f
    }
    LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
    Text(
        text = ModelDownloadUiReducer.waitingMessage(card.downloadState, wifiOnly)
            ?: "${(progress * 100).toInt()}% • ${formatBytes(card.bytesDownloaded)} / ${formatBytes(card.totalBytes)}",
        style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}
```

- [ ] **Step 4: Run full CI-equivalent verification**

```bash
gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace
```

Expected: all tasks PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreen.kt app/src/androidTest/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreenTest.kt
git commit -m "fix: explain model download waiting state"
```

---

### Task 4: Real-device acceptance on the POCO-class target

**Files:**
- No source changes unless the acceptance test exposes a new bug.

**Interfaces:**
- Validates the completed model download behavior against actual Android network classification.

- [ ] **Step 1: Install the CI debug APK over the current debug build**

Use the artifact produced by the branch CI.

- [ ] **Step 2: Verify default behavior**

Open Model Manager and confirm `Wi‑Fiのみでモデルをダウンロード` is OFF by default. Tap UltraSharp `ダウンロード` and verify state reaches RUNNING instead of remaining indefinitely at QUEUED.

- [ ] **Step 3: Verify Wi-Fi-only behavior**

Turn Wi-Fi-only ON. On a network Android classifies as metered/non-unmetered, verify the explanatory waiting message appears. Turn Wi-Fi-only OFF while still waiting and verify the same download restarts and resumes without deleting already-downloaded bytes.

- [ ] **Step 4: Verify model integrity**

Confirm the UltraSharp model reaches installed state only after both artifacts pass expected file-size and SHA-256 checks.

- [ ] **Step 5: Record acceptance result in the next development note/commit message**

No code commit is required when all checks pass; any discovered defect gets its own failing test before a fix.
