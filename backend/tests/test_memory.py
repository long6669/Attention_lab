import unittest

import numpy as np
from app.memory import MemorySpec, MemoryTensorSpec, slice_memory
from app.ops import TensorValue


class TensorPayloadTest(unittest.TestCase):
    def test_large_tensor_payload_omits_values(self) -> None:
        tensor = TensorValue(
            id="large",
            name="Large Tensor",
            value=np.zeros((32, 32), dtype=np.float32),
        )

        payload = tensor.to_dict()

        self.assertFalse(payload["values_loaded"])
        self.assertNotIn("values", payload)
        self.assertEqual(payload["numel"], 1024)
        self.assertEqual(payload["bytes"], 4096)


class MemorySpecTest(unittest.TestCase):
    def test_memory_spec_describes_axes_and_aggregated_blocks(self) -> None:
        cache = np.arange(640, dtype=np.float32).reshape(1, 8, 20, 4)
        spec = MemorySpec(
            kind="attention_cache",
            tensors=(
                MemoryTensorSpec(
                    id="k_cache",
                    name="Key Cache",
                    kind="kv_cache",
                    role="key",
                    value=cache,
                    axes=("batch", "head", "token", "feature"),
                    growth_axis=2,
                ),
            ),
        ).to_dict()

        tensor = spec["tensors"][0]
        self.assertEqual(tensor["growth_axis_name"], "token")
        self.assertFalse(tensor["values_loaded"])
        self.assertNotIn("values", tensor)
        self.assertGreater(len(tensor["blocks"]), 0)
        self.assertLessEqual(len(tensor["blocks"]), 64)

    def test_memory_slice_limits_data_to_requested_head_and_tokens(self) -> None:
        cache = np.arange(160, dtype=np.float32).reshape(1, 2, 20, 4)

        result = slice_memory(
            cache,
            ("batch", "head", "token", "feature"),
            growth_axis=2,
            start=4,
            end=8,
            head=1,
        )

        self.assertEqual(result.shape, (1, 1, 4, 4))
        np.testing.assert_array_equal(result, cache[:, 1:2, 4:8, :])


if __name__ == "__main__":
    unittest.main()
