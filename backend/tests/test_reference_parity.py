import math
import unittest

import numpy as np
from app.architectures import (
    CompressedSparseAttention,
    KimiDeltaAttention,
    MHAConfig,
    MultiHeadAttention,
    MultiHeadLatentAttention,
)


class ReferenceParityTest(unittest.TestCase):
    def test_mha_matches_direct_scaled_dot_product_attention(self) -> None:
        result = MultiHeadAttention().prefill(["one", "two", "three"])
        executor = result.executor
        q = executor.value("q_heads")
        k = result.state.k_cache
        v = result.state.v_cache

        scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(4)
        causal = np.triu(
            np.ones((3, 3), dtype=bool),
            k=1,
        )
        scores = np.where(causal[None, None, :, :], -np.inf, scores)
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        context = np.matmul(probabilities, v)
        expected = np.transpose(context, (0, 2, 1, 3)).reshape(1, 3, 8)

        np.testing.assert_allclose(
            executor.value("attention_probs"),
            probabilities,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            executor.value("output"),
            expected,
            atol=1e-6,
        )

    def test_rope_matches_pairwise_rotation_equation(self) -> None:
        result = MultiHeadAttention(
            MHAConfig(architecture="rope", use_rope=True)
        ).prefill(["one", "two"])
        original = result.executor.value("q_heads")
        rotated = result.executor.value("q_rotated")
        position = 1
        pair = 0
        angle = position / (10000.0 ** (2 * pair / 4))
        even = original[0, 0, position, 0]
        odd = original[0, 0, position, 1]

        expected_even = even * math.cos(angle) - odd * math.sin(angle)
        expected_odd = even * math.sin(angle) + odd * math.cos(angle)

        self.assertAlmostEqual(
            float(rotated[0, 0, position, 0]),
            expected_even,
            places=6,
        )
        self.assertAlmostEqual(
            float(rotated[0, 0, position, 1]),
            expected_odd,
            places=6,
        )

    def test_mla_reconstruction_matches_documented_low_rank_equation(self) -> None:
        result = MultiHeadLatentAttention().prefill(["one", "two"])
        executor = result.executor
        latent = executor.value("latent_cache")
        k_weight = executor.value("k_up_proj_weight")
        v_weight = executor.value("v_up_proj_weight")

        np.testing.assert_allclose(
            executor.value("k_reconstructed"),
            np.matmul(latent, k_weight),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            executor.value("v_reconstructed"),
            np.matmul(latent, v_weight),
            atol=1e-6,
        )

    def test_decode_matches_last_token_of_full_prefill_for_every_path(self) -> None:
        architectures = {
            "mha": MultiHeadAttention(),
            "mqa": MultiHeadAttention(
                MHAConfig(
                    architecture="mqa",
                    num_q_heads=2,
                    num_kv_heads=1,
                )
            ),
            "gqa": MultiHeadAttention(
                MHAConfig(
                    architecture="gqa",
                    num_q_heads=4,
                    num_kv_heads=2,
                )
            ),
            "rope": MultiHeadAttention(MHAConfig(architecture="rope", use_rope=True)),
            "mla": MultiHeadLatentAttention(),
            "kda": KimiDeltaAttention(),
            "csa": CompressedSparseAttention(
                MHAConfig(architecture="csa", routing_top_k=2)
            ),
            "hca": CompressedSparseAttention(
                MHAConfig(architecture="hca", routing_top_k=3)
            ),
        }

        for name, attention in architectures.items():
            with self.subTest(architecture=name):
                prefill = attention.prefill(["one", "two", "three"])
                decode_output = attention.decode(
                    prefill.state,
                    "four",
                ).executor.value("output")
                full_output = attention.prefill(
                    ["one", "two", "three", "four"]
                ).executor.value("output")[:, -1:, :]

                np.testing.assert_allclose(
                    decode_output,
                    full_output,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
