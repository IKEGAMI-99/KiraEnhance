package com.ikegami99.kiraenhance.image

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

sealed interface SaveResult {
    data class Success(val location: String) : SaveResult
    data class Failed(val message: String) : SaveResult
}

object EnhancedImageSaver {
    suspend fun savePng(
        context: Context,
        bitmap: Bitmap,
        displayName: String,
    ): SaveResult = withContext(Dispatchers.IO) {
        require(displayName.endsWith(".png", ignoreCase = true)) { "PNG filename is required" }

        runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                saveWithMediaStore(context, bitmap, displayName)
            } else {
                saveLegacyFallback(context, bitmap, displayName)
            }
        }.fold(
            onSuccess = { SaveResult.Success(it) },
            onFailure = { SaveResult.Failed(it.message ?: "画像を保存できませんでした") },
        )
    }

    private fun saveWithMediaStore(
        context: Context,
        bitmap: Bitmap,
        displayName: String,
    ): String {
        val resolver = context.contentResolver
        val relativePath = "${Environment.DIRECTORY_PICTURES}/KiraEnhance"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, relativePath)
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: error("保存先を作成できませんでした")

        try {
            resolver.openOutputStream(uri, "w")?.use { output ->
                check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)) {
                    "PNGエンコードに失敗しました"
                }
                output.flush()
            } ?: error("保存先を開けませんでした")

            val publishValues = ContentValues().apply {
                put(MediaStore.Images.Media.IS_PENDING, 0)
            }
            resolver.update(uri, publishValues, null, null)
            return "$relativePath/$displayName"
        } catch (error: Throwable) {
            resolver.delete(uri, null, null)
            throw error
        }
    }

    @Suppress("DEPRECATION")
    private fun saveLegacyFallback(
        context: Context,
        bitmap: Bitmap,
        displayName: String,
    ): String {
        val root = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
            ?: context.filesDir
        val directory = File(root, "KiraEnhance").apply { mkdirs() }
        check(directory.isDirectory) { "保存フォルダを作成できませんでした" }
        val file = File(directory, displayName)
        FileOutputStream(file).use { output ->
            check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)) {
                "PNGエンコードに失敗しました"
            }
            output.flush()
        }
        return file.absolutePath
    }
}
