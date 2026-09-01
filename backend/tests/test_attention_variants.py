import unittest

import numpy as np
from app.architectures import MHAConfig, MultiHeadAttention


class MultiQueryAttentionTest(unittest.TestCase):
    def test_mqa_uses_one_kv_head_and_expands_only_for_attention(self) -> None:
        attention = MultiHeadAttention(
            MHAConfig(
                architecture="mqa",
                num_q_heads=2,
                num_kv_heads=1,
            )
        )

        result = attention.prefill(["a", "b", "c"])

        self.assertEqual(result.state.k_cache.shape, (1, 1, 3, 4))
        self.assertEqual(result.state.v_cache.shape, (1, 1, 3, 4))
        self.assertEqual(result.executor.value("k_attention").shape, (1, 2, 3, 4))
        self.assertEqual(result.executor.value("raw_scores").shape, (1, 2, 3, 3))
        self.assertEqual(
            result.executor.value("k_proj_weight").shape,
            (8, 4),
        )
        self.assertEqual(result.to_dict()["memory"]["total_bytes"], 96)

    def test_mqa_decode_keeps_single_kv_head(self) -> None:
        attention = MultiHeadAttention(
            MHAConfig(
                architecture="mqa",
                num_q_heads=2,
                num_kv_heads=1,
            )
        )
        prefill = attention.prefill([f"token_{index}" for index in range(10)])

        decode = attention.decode(prefill.state, "token_11")

        self.assertEqual(prefill.to_dict()["memory"]["total_bytes"], 320)
        self.assertEqual(decode.state.k_cache.shape, (1, 1, 11, 4))
        self.assertEqual(decode.executor.value("q_new_heads").shape, (1, 2, 1, 4))
        self.assertEqual(decode.executor.value("k_new_heads").shape, (1, 1, 1, 4))
        self.assertEqual(decode.executor.value("raw_scores").shape, (1, 2, 1, 11))


class GroupedQueryAttentionTest(unittest.TestCase):
    def test_gqa_maps_two_kv_heads_to_four_query_heads(self) -> None:
        attention = MultiHeadAttention(
            MHAConfig(
                architecture="gqa",
                num_q_heads=4,
                num_kv_heads=2,
            )
        )

        result = attention.prefill(["a", "b", "c"])

        self.assertEqual(result.executor.value("q_heads").shape, (1, 4, 3, 2))
        self.assertEqual(result.state.k_cache.shape, (1, 2, 3, 2))
        self.assertEqual(result.executor.value("k_attention").shape, (1, 4, 3, 2))
        self.assertEqual(result.executor.value("raw_scores").shape, (1, 4, 3, 3))
        self.assertEqual(result.to_dict()["config"]["num_q_heads"], 4)
        self.assertEqual(result.to_dict()["config"]["num_kv_heads"], 2)

    def test_gqa_decode_keeps_grouped_cache_heads(self) -> None:
        attention = MultiHeadAttention(
            MHAConfig(
                architecture="gqa",
                num_q_heads=4,
                num_kv_heads=2,
            )
        )
        prefill = attention.prefill([f"token_{index}" for index in range(10)])

        decode = attention.decode(prefill.state, "token_11")

        self.assertEqual(prefill.to_dict()["memory"]["total_bytes"], 320)
        self.assertEqual(decode.state.k_cache.shape, (1, 2, 11, 2))
        self.assertEqual(decode.executor.value("q_new_heads").shape, (1, 4, 1, 2))
        self.assertEqual(decode.executor.value("k_new_heads").shape, (1, 2, 1, 2))
        self.assertEqual(decode.executor.value("raw_scores").shape, (1, 4, 1, 11))


class RotaryPositionEmbeddingTest(unittest.TestCase):
    def test_rope_rotates_q_and_k_before_caching(self) -> None:
        attention = MultiHeadAttention(
            MHAConfig(
                architecture="rope",
                use_rope=True,
            )
        )

        prefill = attention.prefill(["a", "b", "c"])

        ops = [node.op for node in prefill.executor.graph.nodes]
        self.assertEqual(ops.count("rope"), 2)
        np.testing.assert_array_equal(
            prefill.state.k_cache,
            prefill.executor.value("k_rotated"),
        )
        np.testing.assert_allclose(
            np.linalg.norm(prefill.executor.value("q_heads"), axis=-1),
            np.linalg.norm(prefill.executor.value("q_rotated"), axis=-1),
            atol=1e-6,
        )

    def test_decode_rope_uses_new_token_absolute_position(self) -> None:
        attention = MultiHeadAttention(
            MHAConfig(
                architecture="rope",
                use_rope=True,
            )
        )
        prefill = attention.prefill([f"token_{index}" for index in range(10)])

        decode = attention.decode(prefill.state, "token_11")
        q_rope = decode.executor.graph.get_node("q_rope")

        self.assertIsNotNone(q_rope)
        self.assertEqual(q_rope.attrs["positions"], [10])
        self.assertEqual(decode.executor.value("raw_scores").shape, (1, 2, 1, 11))


if __name__ == "__main__":
    unittest.main()
