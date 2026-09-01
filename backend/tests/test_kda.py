import unittest

import numpy as np

from app.architectures import KimiDeltaAttention


class KimiDeltaAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = KimiDeltaAttention()
        self.tokens = [f"token_{index}" for index in range(10)]

    def test_prefill_scans_decay_erase_and_write(self) -> None:
        result = self.attention.prefill(self.tokens)
        payload = result.to_dict()
        ops = [node.op for node in result.executor.graph.nodes]

        self.assertEqual(result.executor.value("output").shape, (1, 10, 8))
        self.assertEqual(result.state.recurrent_state.shape, (1, 2, 4, 4))
        self.assertEqual(payload["memory"]["cache_kind"], "recurrent")
        self.assertEqual(payload["memory"]["total_bytes"], 128)
        self.assertEqual(
            payload["memory"]["spec"]["tensors"][0]["growth_axis"],
            None,
        )
        for op in ("state_init", "decay", "erase", "write", "scan", "state_update"):
            self.assertIn(op, ops)

    def test_decode_reads_and_updates_fixed_size_state(self) -> None:
        prefill = self.attention.prefill(self.tokens)
        previous = prefill.state.recurrent_state.copy()

        decode = self.attention.decode(prefill.state, "token_11")
        payload = decode.to_dict()

        np.testing.assert_array_equal(
            decode.executor.value("state_previous"),
            previous,
        )
        self.assertEqual(decode.executor.value("new_q_heads").shape, (1, 2, 1, 4))
        self.assertEqual(decode.executor.value("output").shape, (1, 1, 8))
        self.assertEqual(decode.state.recurrent_state.shape, previous.shape)
        self.assertFalse(np.array_equal(decode.state.recurrent_state, previous))
        self.assertEqual(payload["memory"]["total_bytes"], 128)
        self.assertEqual(
            payload["cache_activity"]["update_kind"],
            "state_update",
        )


if __name__ == "__main__":
    unittest.main()
