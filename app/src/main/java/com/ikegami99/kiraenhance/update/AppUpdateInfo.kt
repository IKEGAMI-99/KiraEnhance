package com.ikegami99.kiraenhance.update

data class AppUpdateInfo(
    val versionName: String,
    val versionCode: Int,
    val releaseNotes: String,
    val apkUrl: String,
    val apkSha256: String,
)
