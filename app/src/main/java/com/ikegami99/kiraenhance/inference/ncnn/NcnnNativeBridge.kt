package com.ikegami99.kiraenhance.inference.ncnn

object NcnnNativeBridge {
    init {
        System.loadLibrary("kiraenhance")
    }

    private external fun nativeRuntimeInfo(): String

    fun runtimeInfo(): NcnnRuntimeInfo =
        NcnnRuntimeInfo.parse(nativeRuntimeInfo())
}
