package com.ikegami99.kiraenhance.ui.enhance

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.requiredWidth
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp

@Composable
fun CompareViewer(
    original: ImageBitmap,
    enhanced: ImageBitmap,
    modifier: Modifier = Modifier,
) {
    var dividerFraction by remember { mutableFloatStateOf(0.5f) }
    var zoom by remember { mutableFloatStateOf(1f) }
    var pan by remember { mutableStateOf(Offset.Zero) }

    BoxWithConstraints(
        modifier = modifier
            .clipToBounds()
            .background(MaterialTheme.colorScheme.surfaceContainerHighest)
            .pointerInput(Unit) {
                detectTransformGestures { _, panChange, zoomChange, _ ->
                    val newZoom = (zoom * zoomChange).coerceIn(MIN_ZOOM, MAX_ZOOM)
                    zoom = newZoom
                    pan = if (newZoom <= 1.001f) Offset.Zero else pan + panChange
                }
            },
    ) {
        val containerWidth = maxWidth
        val containerWidthPx = constraints.maxWidth
        val transformModifier = Modifier.graphicsLayer {
            scaleX = zoom
            scaleY = zoom
            translationX = pan.x
            translationY = pan.y
        }

        Image(
            bitmap = original,
            contentDescription = "元画像",
            contentScale = ContentScale.Fit,
            modifier = Modifier.fillMaxSize().then(transformModifier),
        )

        Box(
            modifier = Modifier
                .fillMaxHeight()
                .width(containerWidth * dividerFraction)
                .clipToBounds(),
        ) {
            Image(
                bitmap = enhanced,
                contentDescription = "高画質化後",
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .requiredWidth(containerWidth)
                    .fillMaxHeight()
                    .then(transformModifier),
            )
        }

        Text(
            text = "ORIGINAL",
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.align(Alignment.TopStart),
        )
        Text(
            text = "ENHANCED",
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.align(Alignment.TopEnd),
        )

        Box(
            modifier = Modifier
                .fillMaxHeight()
                .width(18.dp)
                .offset(x = containerWidth * dividerFraction - 9.dp)
                .pointerInput(containerWidthPx) {
                    detectHorizontalDragGestures { change, dragAmount ->
                        change.consume()
                        if (containerWidthPx > 0) {
                            dividerFraction = (
                                dividerFraction + dragAmount / containerWidthPx.toFloat()
                            ).coerceIn(MIN_DIVIDER, MAX_DIVIDER)
                        }
                    }
                },
        ) {
            Box(
                modifier = Modifier
                    .align(Alignment.Center)
                    .fillMaxHeight()
                    .width(2.dp)
                    .background(MaterialTheme.colorScheme.primary),
            )
        }
    }
}

private const val MIN_ZOOM = 1f
private const val MAX_ZOOM = 8f
private const val MIN_DIVIDER = 0.05f
private const val MAX_DIVIDER = 0.95f
