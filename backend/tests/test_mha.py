import unittest

import numpy as np
from app.architectures import MultiHeadAttention


class MultiHeadAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = ["I", "love", "attention"]
        self.run = MultiHeadAttention().run(self.tokens)
        self.executor = self.run.executor

    def test_produces_expected_tensor_shapes(self) -> None:
        expected_shapes = {
            "x": (1, 3, 8),
            "q": (1, 3, 8),
            "k": (1, 3, 8),
            "v": (1, 3, 8),
            "q_heads": (1, 2, 3, 4),
            "k_cache": (1, 2, 3, 4),
            "v_cache": (1, 2, 3, 4),
            "raw_scores": (1, 2, 3, 3),
            "attention_probs": (1, 2, 3, 3),
            "output": (1, 3, 8),
        }

        for tensor_id, shape in expected_shapes.items():
            with self.subTest(tensor=tensor_id):
                self.assertEqual(self.executor.value(tensor_id).shape, shape)

    def test_softmax_rows_sum_to_one(self) -> None:
        probabilities = self.executor.value("attention_probs")

        np.testing.assert_allclose(
            probabilities.sum(axis=-1),
            np.ones((1, 2, 3), dtype=np.float32),
            atol=1e-6,
        )

    def test_causal_mask_removes_future_attention(self) -> None:
        probabilities = self.executor.value("attention_probs")
        future = probabilities[
            ..., np.triu_indices(3, k=1)[0], np.triu_indices(3, k=1)[1]
        ]

        np.testing.assert_array_equal(
            future,
            np.zeros((1, 2, 3), dtype=np.float32),
        )

    def test_embedding_is_deterministic_per_token(self) -> None:
        repeated = MultiHeadAttention().run(["same", "same"])
        other_run = MultiHeadAttention().run(["same"])

        np.testing.assert_array_equal(
            repeated.executor.value("x")[0, 0],
            repeated.executor.value("x")[0, 1],
        )
        np.testing.assert_array_equal(
            repeated.executor.value("x")[0, 0],
            other_run.executor.value("x")[0, 0],
        )

    def test_graph_and_trace_cover_every_primitive(self) -> None:
        graph = self.executor.graph
        trace = self.executor.recorder.events

        self.assertEqual(len(trace), len(graph.nodes))
        self.assertEqual(
            [event.step for event in trace],
            list(range(len(trace))),
        )
        self.assertEqual(trace[0].node_id, "input")
        self.assertEqual(trace[-1].node_id, "output")
        self.assertGreater(len(graph.edges), 0)

    def test_linear_nodes_expose_projection_parameters(self) -> None:
        q_node = self.executor.graph.get_node("q_proj")

        self.assertIsNotNone(q_node)
        self.assertEqual(q_node.attrs["parameter_ids"], ["q_proj_weight"])
        self.assertEqual(
            self.executor.tensors["q_proj_weight"].name,
            "Wq",
        )
        self.assertEqual(
            self.executor.value("q_proj_weight").shape,
            (8, 8),
        )

    def test_memory_calculation_uses_both_float32_caches(self) -> None:
        memory = self.run.to_dict()["memory"]

        self.assertEqual(memory["k_cache"]["shape"], [1, 2, 3, 4])
        self.assertEqual(memory["v_cache"]["shape"], [1, 2, 3, 4])
        self.assertEqual(memory["total_elements"], 48)
        self.assertEqual(memory["total_bytes"], 192)

    def test_payload_replaces_infinity_with_json_null(self) -> None:
        masked_values = self.run.to_dict()["tensors"]["masked_scores"]["values"]

        self.assertIsNone(masked_values[0][0][0][1])
        self.assertIsInstance(masked_values[0][0][0][0], float)


class MultiHeadAttentionDecodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = MultiHeadAttention()
        self.tokens = [f"token_{index}" for index in range(1, 11)]
        self.prefill = self.attention.prefill(self.tokens)
        self.old_k = self.prefill.state.k_cache.copy()
        self.old_v = self.prefill.state.v_cache.copy()
        self.decode = self.attention.decode(
            self.prefill.state,
            "token_11",
        )

    def test_decode_only_projects_the_new_token(self) -> None:
        executor = self.decode.executor

        self.assertEqual(executor.value("x_new").shape, (1, 1, 8))
        self.assertEqual(executor.value("q_new_heads").shape, (1, 2, 1, 4))
        self.assertEqual(executor.value("k_new_heads").shape, (1, 2, 1, 4))
        self.assertEqual(executor.value("v_new_heads").shape, (1, 2, 1, 4))

        linear_nodes = [node for node in executor.graph.nodes if node.op == "linear"]
        self.assertEqual(
            [node.id for node in linear_nodes],
            ["q_new_proj", "k_new_proj", "v_new_proj"],
        )
        self.assertTrue(
            all(node.attrs["input_shapes"] == [[1, 1, 8]] for node in linear_nodes)
        )
        self.assertEqual(
            linear_nodes[0].attrs["parameter_ids"],
            ["q_new_proj_weight"],
        )

    def test_decode_reads_and_preserves_historical_cache(self) -> None:
        executor = self.decode.executor

        np.testing.assert_array_equal(
            executor.value("k_cache_previous"),
            self.old_k,
        )
        np.testing.assert_array_equal(
            executor.value("v_cache_previous"),
            self.old_v,
        )
        np.testing.assert_array_equal(
            self.decode.state.k_cache[..., :10, :],
            self.old_k,
        )
        np.testing.assert_array_equal(
            self.decode.state.v_cache[..., :10, :],
            self.old_v,
        )
        np.testing.assert_array_equal(
            self.decode.state.k_cache[..., 10:, :],
            executor.value("k_new_heads"),
        )
        np.testing.assert_array_equal(
            self.decode.state.v_cache[..., 10:, :],
            executor.value("v_new_heads"),
        )

    def test_decode_shapes_and_memory_match_acceptance_case(self) -> None:
        payload = self.decode.to_dict()

        self.assertEqual(self.decode.state.k_cache.shape, (1, 2, 11, 4))
        self.assertEqual(self.decode.state.v_cache.shape, (1, 2, 11, 4))
        self.assertEqual(
            self.decode.executor.value("raw_scores").shape,
            (1, 2, 1, 11),
        )
        self.assertEqual(payload["memory"]["total_bytes"], 704)
        self.assertEqual(
            payload["cache_activity"],
            {
                "phase": "decode",
                "read_tokens": 10,
                "appended_tokens": 1,
                "resulting_tokens": 11,
            },
        )

    def test_decode_graph_contains_cache_read_and_append(self) -> None:
        ops = [node.op for node in self.decode.executor.graph.nodes]

        self.assertEqual(ops.count("cache_read"), 2)
        self.assertEqual(ops.count("cache_append"), 2)
        self.assertLess(ops.index("cache_read"), ops.index("cache_append"))


if __name__ == "__main__":
    unittest.main()
