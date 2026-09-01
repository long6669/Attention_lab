import unittest

import numpy as np
from app.architectures import MultiHeadLatentAttention


class MultiHeadLatentAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = MultiHeadLatentAttention()
        self.tokens = [f"token_{index}" for index in range(1, 11)]
        self.prefill = self.attention.prefill(self.tokens)

    def test_prefill_compresses_kv_into_latent_cache(self) -> None:
        payload = self.prefill.to_dict()

        self.assertEqual(
            self.prefill.executor.value("kv_latent").shape,
            (1, 10, 4),
        )
        self.assertEqual(self.prefill.state.latent_cache.shape, (1, 10, 4))
        self.assertEqual(
            self.prefill.executor.value("raw_scores").shape,
            (1, 2, 10, 10),
        )
        self.assertEqual(payload["memory"]["cache_kind"], "latent")
        self.assertEqual(payload["memory"]["total_bytes"], 160)
        self.assertEqual(
            [
                node.op
                for node in self.prefill.executor.graph.nodes
                if node.op == "low_rank_compression"
            ],
            ["low_rank_compression"],
        )

    def test_decode_appends_only_one_latent_vector(self) -> None:
        previous = self.prefill.state.latent_cache.copy()

        decode = self.attention.decode(self.prefill.state, "token_11")
        payload = decode.to_dict()

        self.assertEqual(
            decode.executor.value("kv_latent_new").shape,
            (1, 1, 4),
        )
        self.assertEqual(decode.state.latent_cache.shape, (1, 11, 4))
        np.testing.assert_array_equal(
            decode.state.latent_cache[:, :10],
            previous,
        )
        np.testing.assert_array_equal(
            decode.state.latent_cache[:, 10:],
            decode.executor.value("kv_latent_new"),
        )
        self.assertEqual(
            decode.executor.value("raw_scores").shape,
            (1, 2, 1, 11),
        )
        self.assertEqual(payload["memory"]["total_bytes"], 176)
        self.assertNotIn("k_new", decode.executor.tensors)
        self.assertNotIn("v_new", decode.executor.tensors)


if __name__ == "__main__":
    unittest.main()
