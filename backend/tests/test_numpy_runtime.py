import unittest

import numpy as np
from app.runtime import NumPyRuntime


class NumPyRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = NumPyRuntime()

    def test_linear_projects_last_dimension_and_adds_bias(self) -> None:
        inputs = np.array(
            [[[1.0, 2.0], [3.0, 4.0]]],
            dtype=np.float32,
        )
        weight = np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        bias = np.array([0.5, -0.5, 1.0], dtype=np.float32)

        output = self.runtime.linear(inputs, weight, bias)

        expected = np.array(
            [[[1.5, 1.5, 4.0], [3.5, 3.5, 8.0]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(output, expected)
        self.assertEqual(output.shape, (1, 2, 3))
        self.assertEqual(output.dtype, np.float32)

    def test_linear_rejects_incompatible_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "input dimension"):
            self.runtime.linear(np.ones((1, 2, 3)), np.ones((4, 2)))

        with self.assertRaisesRegex(ValueError, "bias"):
            self.runtime.linear(
                np.ones((1, 2, 3)),
                np.ones((3, 2)),
                np.ones(3),
            )

    def test_batched_matmul(self) -> None:
        left = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
        right = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)

        output = self.runtime.matmul(left, right)

        np.testing.assert_allclose(output, np.matmul(left, right))
        self.assertEqual(output.shape, (1, 2, 2, 4))

    def test_reshape_and_transpose_for_attention_heads(self) -> None:
        tensor = np.arange(24, dtype=np.float32).reshape(1, 3, 8)

        split = self.runtime.reshape(tensor, (1, 3, 2, 4))
        transposed = self.runtime.transpose(split, (0, 2, 1, 3))

        self.assertEqual(transposed.shape, (1, 2, 3, 4))
        np.testing.assert_array_equal(transposed[0, 1, 0], [4, 5, 6, 7])

    def test_repeat_kv_maps_kv_heads_to_query_heads(self) -> None:
        cache = np.array(
            [[[[1.0, 2.0]], [[3.0, 4.0]]]],
            dtype=np.float32,
        )

        repeated = self.runtime.repeat_kv(cache, repeats=2)

        self.assertEqual(repeated.shape, (1, 4, 1, 2))
        np.testing.assert_array_equal(repeated[:, 0], cache[:, 0])
        np.testing.assert_array_equal(repeated[:, 1], cache[:, 0])
        np.testing.assert_array_equal(repeated[:, 2], cache[:, 1])
        np.testing.assert_array_equal(repeated[:, 3], cache[:, 1])

    def test_rope_uses_absolute_positions_and_preserves_norm(self) -> None:
        tensor = np.array(
            [[[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]]],
            dtype=np.float32,
        )

        rotated = self.runtime.rope(tensor, positions=[0, 1])

        np.testing.assert_array_equal(rotated[..., 0, :], tensor[..., 0, :])
        self.assertFalse(np.array_equal(rotated[..., 1, :], tensor[..., 1, :]))
        np.testing.assert_allclose(
            np.linalg.norm(rotated, axis=-1),
            np.linalg.norm(tensor, axis=-1),
            atol=1e-6,
        )

    def test_scale_divides_attention_scores(self) -> None:
        scores = np.array([[2.0, 4.0]], dtype=np.float32)

        scaled = self.runtime.scale(scores, divisor=2.0)

        np.testing.assert_allclose(scaled, [[1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "must not be zero"):
            self.runtime.scale(scores, divisor=0)

    def test_causal_mask_hides_future_positions(self) -> None:
        scores = np.zeros((1, 2, 3, 3), dtype=np.float32)

        masked = self.runtime.causal_mask(scores)

        expected_head = np.array(
            [
                [0.0, -np.inf, -np.inf],
                [0.0, 0.0, -np.inf],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(masked[0, 0], expected_head)
        np.testing.assert_array_equal(masked[0, 1], expected_head)

    def test_causal_mask_aligns_short_queries_to_cache_tail(self) -> None:
        scores = np.zeros((1, 2, 1, 4), dtype=np.float32)

        masked = self.runtime.causal_mask(scores)

        np.testing.assert_array_equal(masked, scores)

    def test_softmax_is_stable_and_rows_sum_to_one(self) -> None:
        values = np.array(
            [[1000.0, 1000.0], [1.0, 2.0]],
            dtype=np.float32,
        )

        probabilities = self.runtime.softmax(values)

        np.testing.assert_allclose(
            probabilities.sum(axis=-1),
            np.ones(2, dtype=np.float32),
        )
        np.testing.assert_allclose(probabilities[0], [0.5, 0.5])
        self.assertTrue(np.isfinite(probabilities).all())

    def test_masked_softmax_assigns_zero_to_future_positions(self) -> None:
        scores = np.ones((3, 3), dtype=np.float32)

        probabilities = self.runtime.softmax(self.runtime.causal_mask(scores))

        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
        np.testing.assert_array_equal(
            probabilities[np.triu_indices(3, k=1)],
            np.zeros(3, dtype=np.float32),
        )

    def test_kda_scan_applies_delta_write_and_returns_final_state(self) -> None:
        q = np.array([[[[1.0, 0.0]]]], dtype=np.float32)
        k = np.array([[[[1.0, 0.0]]]], dtype=np.float32)
        v = np.array([[[[2.0, 3.0]]]], dtype=np.float32)
        state = np.zeros((1, 1, 2, 2), dtype=np.float32)

        output, decayed, erased, written, final_state = self.runtime.kda_scan(
            q,
            k,
            v,
            state,
            decay=0.9,
            write_rate=0.5,
        )

        expected = np.array([[[[1.0, 1.5], [0.0, 0.0]]]], dtype=np.float32)
        np.testing.assert_allclose(final_state, expected)
        np.testing.assert_allclose(written, expected[:, :, None, :, :])
        np.testing.assert_allclose(output, [[[[1.0, 1.5]]]])
        np.testing.assert_array_equal(decayed, 0.0)
        np.testing.assert_array_equal(erased, 0.0)

    def test_sequence_compression_builds_causal_window_summaries(self) -> None:
        values = np.array([[[[1.0], [2.0], [4.0]]]], dtype=np.float32)

        compressed, spans = self.runtime.sequence_compress(values, [2])

        self.assertEqual(spans, [(0, 1, 0), (0, 2, 0), (1, 3, 0)])
        np.testing.assert_allclose(
            compressed.reshape(-1),
            [1.0, 1.5, 3.0],
        )

    def test_index_topk_and_routing_keep_selected_context(self) -> None:
        queries = np.ones((1, 1, 3, 2), dtype=np.float32)
        keys = np.array(
            [[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]],
            dtype=np.float32,
        )
        values = np.array(
            [[[[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]]]],
            dtype=np.float32,
        )
        spans = [(0, 1, 0), (0, 2, 0), (1, 3, 0)]

        scores = self.runtime.index_scores(queries, keys, spans, query_offset=0)
        indices = self.runtime.topk(scores, 2)
        selected = self.runtime.route_scores(scores, indices)
        routed = self.runtime.route_values(values, indices)
        probabilities = self.runtime.softmax(selected)
        output = self.runtime.weighted_route(probabilities, routed)

        self.assertTrue(np.isneginf(scores[0, 0, 0, 1:]).all())
        self.assertEqual(indices.shape, (1, 1, 3, 2))
        self.assertEqual(routed.shape, (1, 1, 3, 2, 2))
        self.assertEqual(output.shape, (1, 1, 3, 2))
        self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
