package com.ikegami99.kiraenhance.diagnostics

import com.ikegami99.kiraenhance.inference.mnn.MnnRuntimeInfo

object MnnRuntimeDiagnostics {
    fun capture(probe: () -> MnnRuntimeInfo): String =
        runCatching(probe).fold(
            onSuccess = { info ->
                "version=${info.mnnVersion} linked=${info.linked}"
            },
            onFailure = { error ->
                "probe_failed type=${error.javaClass.simpleName} message=${error.message ?: "no-message"}"
            },
        )
}
