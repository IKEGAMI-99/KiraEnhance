# KiraEnhance Model Download and In-App Update Design

## Scope

This change has two goals:

1. Fix model downloads that can remain waiting in WorkManager and only show `ダウンロード待機中`.
2. Add an in-app update flow so future KiraEnhance APK updates can be discovered and installed from inside the app.

The existing HTTP 416 recovery remains in place.

## Model Download Design

### Current problem

`ModelManagerUiState` defaults to `wifiOnly = true`, which maps to `NetworkType.UNMETERED`. On some Android network configurations, including VPN or metered-classified Wi-Fi, WorkManager may keep the request `ENQUEUED` while its network constraint is unsatisfied even though the device is visibly connected to Wi-Fi. The UI currently renders both `ENQUEUED` and `BLOCKED` as the same generic `ダウンロード待機中` label, so the user cannot tell whether KiraEnhance is waiting on the Wi-Fi-only constraint or on another WorkManager prerequisite.

### New behavior

- Default `wifiOnly` to `false`.
- With `wifiOnly = false`, use `NetworkType.CONNECTED`.
- With `wifiOnly = true`, keep `NetworkType.UNMETERED`.
- Distinguish `ENQUEUED` from `BLOCKED` in UI state.
- When work is `ENQUEUED` while Wi-Fi-only is enabled, show `Wi‑Fiのみ設定のため、非従量制ネットワークを待っています`.
- When work is `ENQUEUED` with Wi-Fi-only disabled, show `ネットワーク接続または実行開始を待っています`.
- When work is `BLOCKED`, show `前提となる処理の完了を待っています`.
- Allow the user to switch off Wi-Fi-only mode while a model is queued/blocked and immediately replace that unique work request using `ExistingWorkPolicy.REPLACE` without deleting the partial model files.
- Ignore stale WorkInfo callbacks from a replaced request so an old cancellation event cannot overwrite the state of the new request.
- Keep existing progress, checksum verification, partial-file resume, and HTTP 416 restart-from-zero handling.

## In-App Update Design

### Distribution model

Use GitHub Releases as the source of truth for app updates. Each release provides:

- semantic version name
- monotonically increasing Android `versionCode`
- signed APK asset
- SHA-256 checksum for the APK

The release also contains an `update-manifest.json` asset at the stable GitHub Releases `latest/download` URL. The manifest contains `versionName`, `versionCode`, `releaseNotes`, `apkUrl`, and `apkSha256`. The app downloads this small manifest over HTTPS and compares its `versionCode` with the installed app version.

### Signing

All updateable APKs must use one stable release signing key. The current debug-build workflow cannot be used for long-term in-place updates because mismatched debug signing breaks Android package replacement.

The release signing keystore and passwords will live in GitHub Actions Secrets. CI will produce a release-signed APK only when the required secrets are present. Existing debug APKs remain for development and testing.

The first stable-signed build may require one manual uninstall/reinstall. After that, updates can install over the existing app as long as application ID and signing certificate remain unchanged.

### App-side updater

Add an `update` package with small, testable components:

- `AppUpdateChecker`: fetches and parses `update-manifest.json` and compares version codes.
- `AppUpdateInfo`: version name, version code, release notes, APK URL, and APK SHA-256.
- `AppUpdateDownloadManager` / `AppUpdateDownloadWorker`: downloads the APK to app-managed storage, verifies SHA-256, and exposes progress.
- `AppUpdateInstaller`: launches Android's package installer using a `FileProvider` content URI.

### Permissions and Android integration

- Keep `INTERNET`.
- Add `REQUEST_INSTALL_PACKAGES` so the app can request package installation from its own downloaded APK.
- Add a `FileProvider` with a narrow path limited to the update APK directory.
- If install-from-this-source permission is disabled, open Android's `ACTION_MANAGE_UNKNOWN_APP_SOURCES` settings page for KiraEnhance.
- Never silently install. Android's system installer confirmation remains mandatory.

### UI

There is no settings/about screen in the current app, so add a dedicated `アプリ更新` screen reachable from Home. It contains:

- current version
- `アップデートを確認` button
- latest version and release notes when available
- download progress
- `インストール` button after checksum verification
- clear error text for network, parsing, checksum, permission, and installer-launch failures

For alpha builds, update checks are manual only. Automatic background polling is out of scope.

## Versioning and Release Workflow

- Keep explicit Android `versionCode` and `versionName`, but allow the release workflow to provide them through build environment variables instead of editing source for each release.
- Add a GitHub Actions release workflow that builds a release APK, signs it with the stable keystore, computes SHA-256, generates `update-manifest.json`, and attaches the APK, checksum, and manifest to a GitHub Release.
- Development CI continues building debug APKs.
- Required GitHub Secrets are `KIRA_RELEASE_KEYSTORE_B64`, `KIRA_RELEASE_STORE_PASSWORD`, `KIRA_RELEASE_KEY_ALIAS`, and `KIRA_RELEASE_KEY_PASSWORD`.

## Error Handling

Model downloads:

- `ENQUEUED` + Wi-Fi-only: show the non-metered-network explanation.
- `ENQUEUED` without Wi-Fi-only: show a generic network/start wait explanation.
- `BLOCKED`: show a WorkManager prerequisite explanation.
- HTTP 416 during resume: delete partial file and retry from zero in the same worker.
- checksum mismatch: delete the bad artifact and fail visibly.

App updates:

- update manifest unavailable: keep the installed app usable and show an error.
- APK download interrupted: resume where safe; restart from zero on invalid range state.
- checksum mismatch: delete the APK and never offer install.
- unknown-app-sources permission disabled: send the user to the system settings page for KiraEnhance.
- signature/package mismatch: Android installer error remains visible to the user; no bypass attempt.

## Testing

Use TDD for behavior changes.

Minimum tests:

- default model network policy is `CONNECTED`.
- Wi-Fi-only policy is `UNMETERED`.
- `ENQUEUED` and `BLOCKED` map to distinct UI states/messages.
- switching Wi-Fi-only off for queued/blocked work replaces the existing unique work without deleting the partial files.
- stale callbacks from replaced work are ignored.
- HTTP 416 resume regression test remains green.
- update version comparison tests.
- update-manifest parsing tests.
- APK SHA-256 verification tests.
- installer intent construction tests where practical.
- full Android CI build, unit tests, and androidTest compilation must pass before publishing an APK.

## Non-Goals

- Play Store / Play In-App Updates integration.
- silent or unattended APK installation.
- background automatic update checks.
- self-hosted update server.
- model-update catalog versioning beyond the existing model manifest.
