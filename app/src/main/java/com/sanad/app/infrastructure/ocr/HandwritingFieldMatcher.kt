package com.sanad.app.infrastructure.ocr

import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sin

/** Geometry shared by the form-specific anchor and shared handwriting detectors. */
internal data class OrientedBox(
    val centerX: Float,
    val centerY: Float,
    val width: Float,
    val height: Float,
    val angle: Float,
    val confidence: Float,
    val classId: Int = 0,
)

internal data class AnchoredFieldRegion(
    val fieldId: String,
    val anchor: OrientedBox,
    /** Anchor-local x1, y1, x2, y2. The detected anchor occupies 0..1 on both axes. */
    val region: FloatArray,
)

/**
 * Assigns shared handwriting detections to form fields one-to-one.
 *
 * Relative regions constrain association only. OCR crops always come from the matched handwriting OBB,
 * so errors in the printed-label box do not directly become errors in the final crop.
 */
internal object HandwritingFieldMatcher {
    fun match(fields: List<AnchoredFieldRegion>, handwriting: List<OrientedBox>): Map<String, OrientedBox> {
        if (fields.isEmpty() || handwriting.isEmpty()) return emptyMap()
        val columnCount = handwriting.size + fields.size
        val costs = Array(fields.size) { FloatArray(columnCount) { UNASSIGNED_COST } }
        fields.forEachIndexed { fieldIndex, field ->
            handwriting.forEachIndexed { detectionIndex, detection ->
                costs[fieldIndex][detectionIndex] = associationCost(field, detection)
            }
        }
        val assignment = hungarian(costs)
        return buildMap {
            assignment.forEachIndexed { fieldIndex, column ->
                if (column in handwriting.indices && costs[fieldIndex][column] < UNASSIGNED_COST) {
                    put(fields[fieldIndex].fieldId, handwriting[column])
                }
            }
        }
    }

    private fun associationCost(field: AnchoredFieldRegion, detection: OrientedBox): Float {
        val anchor = field.anchor
        val cosine = cos(anchor.angle)
        val sine = sin(anchor.angle)
        val dx = detection.centerX - anchor.centerX
        val dy = detection.centerY - anchor.centerY
        val localX = 0.5f + (dx * cosine + dy * sine) / max(anchor.width, 1f)
        val localY = 0.5f + (-dx * sine + dy * cosine) / max(anchor.height, 1f)
        val x1 = field.region[0]
        val y1 = field.region[1]
        val x2 = field.region[2]
        val y2 = field.region[3]
        val regionWidth = max(x2 - x1, 0.2f)
        val regionHeight = max(y2 - y1, 0.5f)
        val outsideX = max(x1 - localX, 0f) + max(localX - x2, 0f)
        val outsideY = max(y1 - localY, 0f) + max(localY - y2, 0f)
        val outside = hypot(outsideX / regionWidth, outsideY / regionHeight)
        if (outside > MAX_OUTSIDE_DISTANCE) return REJECTED_COST

        val expectedX = (x1 + x2) * 0.5f
        val expectedY = (y1 + y2) * 0.5f
        val centerDistance = hypot((localX - expectedX) / regionWidth, (localY - expectedY) / regionHeight)
        val orientationDifference = kotlin.math.abs(normalizeAngle(detection.angle - anchor.angle))
        return outside * 3f + centerDistance * 0.35f + orientationDifference * 0.08f + (1f - detection.confidence) * 0.15f
    }

    /** Rectangular Hungarian assignment with one private dummy column per field. */
    private fun hungarian(cost: Array<FloatArray>): IntArray {
        val rows = cost.size
        val columns = cost.first().size
        val rowPotential = FloatArray(rows + 1)
        val columnPotential = FloatArray(columns + 1)
        val matchedRow = IntArray(columns + 1)
        val path = IntArray(columns + 1)
        for (row in 1..rows) {
            matchedRow[0] = row
            var currentColumn = 0
            val minimum = FloatArray(columns + 1) { Float.POSITIVE_INFINITY }
            val used = BooleanArray(columns + 1)
            do {
                used[currentColumn] = true
                val currentRow = matchedRow[currentColumn]
                var delta = Float.POSITIVE_INFINITY
                var nextColumn = 0
                for (column in 1..columns) if (!used[column]) {
                    val reduced = cost[currentRow - 1][column - 1] - rowPotential[currentRow] - columnPotential[column]
                    if (reduced < minimum[column]) {
                        minimum[column] = reduced
                        path[column] = currentColumn
                    }
                    if (minimum[column] < delta) {
                        delta = minimum[column]
                        nextColumn = column
                    }
                }
                for (column in 0..columns) {
                    if (used[column]) {
                        rowPotential[matchedRow[column]] += delta
                        columnPotential[column] -= delta
                    } else minimum[column] -= delta
                }
                currentColumn = nextColumn
            } while (matchedRow[currentColumn] != 0)
            do {
                val previousColumn = path[currentColumn]
                matchedRow[currentColumn] = matchedRow[previousColumn]
                currentColumn = previousColumn
            } while (currentColumn != 0)
        }
        return IntArray(rows) { -1 }.also { result ->
            for (column in 1..columns) if (matchedRow[column] > 0) result[matchedRow[column] - 1] = column - 1
        }
    }

    private fun normalizeAngle(value: Float): Float {
        var angle = value
        while (angle > Math.PI / 2) angle -= Math.PI.toFloat()
        while (angle < -Math.PI / 2) angle += Math.PI.toFloat()
        return angle
    }

    private const val UNASSIGNED_COST = 2.5f
    private const val REJECTED_COST = 10f
    private const val MAX_OUTSIDE_DISTANCE = 0.65f
}
