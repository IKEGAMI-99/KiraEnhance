package com.ikegami99.kiraenhance.ui.smoke

import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeImage
import java.io.IOException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class NcnnSmokeSessionTest {
    private val input = NcnnSmokeImage(1, 1, intArrayOf(0xff123456.toInt()))
    private val output = NcnnSmokeImage(4, 4, IntArray(16) { 0xff123456.toInt() })

    @Test fun selectionAndInferencePublishPixelsAndDimensions() = runBlocking {
        val session = session()
        session.selectImage("content://local/image")
        assertSame(input, session.state.value.input)
        assertTrue(session.state.value.screen.imageSelected)
        session.run()
        assertSame(output, session.state.value.output)
        assertEquals("4 × 4", session.state.value.screen.outputDescription)
        assertFalse(session.state.value.screen.isRunning)
    }

    @Test fun pickerCancellationPreservesResultAndNewSelectionClearsIt() = runBlocking {
        val session = session()
        session.selectImage("content://local/one")
        session.run()
        session.selectImage(null)
        assertSame(output, session.state.value.output)
        session.selectImage("content://local/two")
        assertNull(session.state.value.output)
        assertNull(session.state.value.screen.outputDescription)
        assertNull(session.state.value.screen.elapsedMs)
    }

    @Test fun unreadableImageClearsStalePixelsAndDoesNotExposeUri() = runBlocking {
        val session = session(decode = { uri ->
            if (uri.endsWith("private")) throw IOException(uri)
            input
        })
        session.selectImage("content://local/good")
        session.run()
        session.selectImage("content://local/private")
        assertNull(session.state.value.input)
        assertNull(session.state.value.output)
        assertFalse(session.state.value.screen.imageSelected)
        assertFalse(session.state.value.screen.isRunning)
        assertFalse(session.state.value.screen.status.contains("content://"))
    }

    @Test fun runningInferenceRejectsDuplicateRunAndImageReplacement() = runBlocking {
        val finish = CompletableDeferred<Unit>()
        var calls = 0
        val session = session(infer = {
            calls++
            finish.await()
            completed()
        })
        session.selectImage("content://local/one")
        val job = launch(Dispatchers.Unconfined) { session.run() }
        assertTrue(session.state.value.screen.isRunning)
        session.run()
        session.selectImage("content://local/two")
        assertEquals(1, calls)
        assertSame(input, session.state.value.input)
        finish.complete(Unit)
        job.join()
        assertSame(output, session.state.value.output)
    }

    @Test fun missingModelOrImageCannotStartInference() = runBlocking {
        var calls = 0
        val session = session(infer = { calls++; completed() })
        session.run()
        session.selectImage("content://local/one")
        session.refreshModelInstallation(false)
        session.run()
        assertEquals(0, calls)
        assertFalse(session.state.value.screen.modelInstalled)
    }

    @Test fun cancelledOperationRestoresIdleStateAndRethrowsCancellation() = runBlocking {
        val started = CompletableDeferred<Unit>()
        val session = session(infer = {
            started.complete(Unit)
            CompletableDeferred<Unit>().await()
            completed()
        })
        session.selectImage("content://local/one")
        val job = launch { session.run() }
        started.await()
        job.cancel()
        job.join()
        assertTrue(job.isCancelled)
        assertFalse(session.state.value.screen.isRunning)
        assertNull(session.state.value.output)
    }

    private fun completed() = NcnnSmokePresentationResult(
        NcnnSmokeScreenState(true, "complete", imageSelected = true,
            inputDescription = "1 × 1", outputDescription = "4 × 4", elapsedMs = 5),
        output,
    )

    private fun session(
        decode: suspend (String) -> NcnnSmokeImage = { input },
        infer: suspend (NcnnSmokeImage) -> NcnnSmokePresentationResult = { completed() },
    ) = NcnnSmokeSession(
        initialState = NcnnSmokeScreenState(true, "ready"),
        decode = decode,
        infer = infer,
        workerDispatcher = Dispatchers.Unconfined,
    )
}

