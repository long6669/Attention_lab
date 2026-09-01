import unittest

import numpy as np

from app.architectures import CompressedSparseAttention, MHAConfig


class CompressedSparseAttentionTest(unittest.TestCase):
    def test_csa_compresses_indexes_and_routes_top_k(self) -> None:
        attention = CompressedSparseAttention(
            MHAConfig(
                architecture="csa",
                routing_top_k=2,
            )
        )

        result = attention.prefill(["a", "b", "c", "d"])
        ops = [node.op for node in result.executor.graph.nodes]

        self.assertEqual(result.executor.value("compressed_k").shape, (1, 2, 4, 4))
        self.assertEqual(result.executor.value("route_indices").shape, (1, 2, 4, 2))
        self.assertEqual(result.executor.value("routing_probs").shape, (1, 2, 4, 2))
        self.assertEqual(result.executor.value("output").shape, (1, 4, 8))
        np.testing.assert_allclose(
            result.executor.value("routing_probs").sum(axis=-1),
            1.0,
        )
        for op in ("sequence_compression", "indexer", "topk", "routing"):
            self.assertIn(op, ops)

    def test_hca_builds_two_compression_levels(self) -> None:
        attention = CompressedSparseAttention(
            MHAConfig(
                architecture="hca",
                routing_top_k=3,
            )
        )

        result = attention.prefill(["a", "b", "c", "d"])
        compression = result.executor.graph.get_node("sequence_compress_k")

        self.assertIsNotNone(compression)
        self.assertEqual(compression.attrs["window_sizes"], [2, 4])
        self.assertEqual(result.executor.value("compressed_k").shape, (1, 2, 8, 4))
        self.assertEqual(result.executor.value("route_indices").shape, (1, 2, 4, 3))

    def test_sparse_decode_routes_new_query_over_updated_cache(self) -> None:
        attention = CompressedSparseAttention(
            MHAConfig(
                architecture="csa",
                routing_top_k=2,
            )
        )
        prefill = attention.prefill([f"token_{index}" for index in range(10)])

        decode = attention.decode(prefill.state, "token_11")
        indexer = decode.executor.graph.get_node("indexer")

        self.assertEqual(decode.state.k_cache.shape, (1, 2, 11, 4))
        self.assertEqual(decode.executor.value("new_q_heads").shape, (1, 2, 1, 4))
        self.assertEqual(decode.executor.value("routing_scores").shape, (1, 2, 1, 11))
        self.assertEqual(decode.executor.value("output").shape, (1, 1, 8))
        self.assertEqual(indexer.attrs["query_offset"], 10)


if __name__ == "__main__":
    unittest.main()
