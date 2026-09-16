# KiraEnhance Model Download and In-App Update Design

## Scope

This change has two goals:

1. Fix model downloads that can remain stuck in `ENQUEUED/BLOCKED` and only show `ダウンロード待機中`.
2. Add an in-app update flow so future KiraEnhance APK updates can be discovered and installed from inside the app.

The existing HTTP 416 recovery remains in place.

## Model Download Design

### Current problem

`ModelDownloadManager` defaults to `wifiOnly = true`, which maps to `NetworkType.UNMETERED`. On some Android network configurations, including VPN or metered-classified Wi-Fi, WorkManager may keep the request blocked even though the device is visibly connected to Wi-Fi. The UI currently maps both `ENQUEUED` and `BLOCKED` to the same generic `ダウンロード待機中` label, so the reason is hidden.

### New behavior

- Default `wifiOnly` to `false`.
- With `wifiOnly = false`, use `NetworkType.CONNECTED`.
- With `wifiOnly = true`, keep `NetworkType.UNMETERED`.
- Distinguish `ENQUEUED` from `BLOCKED` in UI state.
- For blocked work, show a reason-oriented message such as `Wi‑Fiのみ設定のため、非従量制ネットワークを待っています`.
- Allow the user to switch off Wi-Fi-only mode while a model is queued/blocked and immediately replace that work request using `ExistingWorkPolicy.REPLACE`.
- Keep existing progress, checksum verification, partial-file resume, and HTTP 416 restart-from-zero handling.

## In-App Update Design

### Distribution model

Use GitHub Releases as the source of truth for app updates. Each release provides:

- semantic version name
- monotonically increasing Android `versionCode`
- signed APK asset
- SHA-256 checksum for the APK

The app checks the latest GitHub Release metadata over HTTPS and compares its release version with the installed app version.

### Signing

All updateable APKs must use one stable release signing key. The current debug-build workflow cannot be used for long-term in-place updates because ephemeral or mismatched debug signing breaks Android package replacement.

The release signing keystore and passwords will live in GitHub Actions Secrets. CI will produce a release-signed APK only when the required secrets are present. Existing debug APKs remain for development and testing.

The first stable-signed build may require one manual uninstall/reinstall. After that, updates can install over the existing app as long as application ID and signing certificate remain unchanged.

### App-side updater

Add an `update` package with small, testable components:

- `AppUpdateChecker`: fetches and parses release metadata.
- `AppUpdateInfo`: version, release notes, APK URL, APK SHA-256, required version code.
- `AppUpdateDownloadManager` / worker: downloads the APK to app-managed storage, verifies SHA-256, and exposes progress.
- `AppUpdateInstaller`: launches Android's package installer using a `FileProvider` content URI.

### Permissions and Android integration

- Keep `INTERNET`.
- Add `REQUEST_INSTALL_PACKAGES` so the app can request package installation from its own downloaded APK.
- Add a `FileProvider` with a narrow path limited to the update APK directory.
- If install-from-this-source permission is disabled, open Android's `ACTION_MANAGE_UNKNOWN_APP_SOURCES` settings page for KiraEnhance.
- Never silently install. Android's system installer confirmation remains mandatory.

### UI

Add an `アプリ更新` section in the app settings/about area:

- current version
- `アップデートを確認` button
- latest version and release notes when available
- download progress
- `インストール` button after checksum verification
- clear error text for network, parsing, checksum, permission, and installer-launch failures

For alpha builds, update checks are manual only. Automatic background polling is out of scope.

## Versioning and Release Workflow

- Replace hard-coded version handling with explicit `versionCode` and `versionName` values intended to advance per release.
- Add a GitHub Actions release workflow that builds a release APK, signs it with the stable keystore, computes SHA-256, and attaches both APK and checksum to a GitHub Release.
- Development CI continues building debug APKs.

## Error Handling

Model downloads:

- `BLOCKED`: show network-constraint explanation rather than a generic wait message.
- HTTP 416 during resume: delete partial file and retry from zero in the same worker.
- checksum mismatch: delete the bad artifact and fail visibly.

App updates:

- release metadata unavailable: keep the installed app usable and show an error.
- APK download interrupted: resume where safe; restart from zero on invalid range state.
- checksum mismatch: delete the APK and never offer install.
- signature/package mismatch: Android installer error is surfaced to the user; no bypass attempt.

## Testing

Use TDD for behavior changes.

Minimum tests:

- default model network policy is `CONNECTED`.
- Wi-Fi-only policy is `UNMETERED`.
- blocked WorkInfo maps to a distinct blocked UI state/message.
- switching Wi-Fi-only off for blocked work replaces the existing unique work.
- HTTP 416 resume regression test remains green.
- update version comparison tests.
- release metadata parsing tests.
- APK SHA-256 verification tests.
- installer intent construction tests where practical.
- full Android CI build, unit tests, and androidTest compilation must pass before publishing an APK.

## Non-Goals

- Play Store / Play In-App Updates integration.
- silent or unattended APK installation.
- background automatic update checks.
- self-hosted update server.
- model-update catalog versioning beyond the existing model manifest.
