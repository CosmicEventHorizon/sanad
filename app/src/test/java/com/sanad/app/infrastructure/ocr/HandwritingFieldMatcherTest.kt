package com.sanad.app.infrastructure.ocr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class HandwritingFieldMatcherTest {
    @Test
    fun `matches detected handwriting to nearby anchored fields one to one`() {
        val firstAnchor = box(100f, 100f, 40f, 12f)
        val secondAnchor = box(100f, 180f, 40f, 12f)
        val fields = listOf(
            AnchoredFieldRegion("region", firstAnchor, floatArrayOf(1f, -0.5f, 5f, 1.5f)),
            AnchoredFieldRegion("province", secondAnchor, floatArrayOf(1f, -0.5f, 5f, 1.5f)),
        )
        val firstHandwriting = box(190f, 100f, 120f, 18f, confidence = 0.91f)
        val secondHandwriting = box(185f, 180f, 110f, 18f, confidence = 0.88f)

        val matches = HandwritingFieldMatcher.match(fields, listOf(secondHandwriting, firstHandwriting))

        assertEquals(firstHandwriting, matches["region"])
        assertEquals(secondHandwriting, matches["province"])
    }

    @Test
    fun `rejects handwriting outside the label relative search region`() {
        val fields = listOf(
            AnchoredFieldRegion("region", box(100f, 100f, 40f, 12f), floatArrayOf(1f, -0.5f, 5f, 1.5f)),
        )

        val matches = HandwritingFieldMatcher.match(fields, listOf(box(100f, 300f, 120f, 18f)))

        assertFalse(matches.containsKey("region"))
    }

    private fun box(
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        angle: Float = 0f,
        confidence: Float = 0.9f,
    ) = OrientedBox(centerX, centerY, width, height, angle, confidence)
}
