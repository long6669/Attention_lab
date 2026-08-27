import unittest

from app.ir import Edge, Graph, Node, TensorSpec


class TensorSpecTest(unittest.TestCase):
    def test_serializes_shape_as_json_array(self) -> None:
        tensor = TensorSpec(
            id="tensor_q",
            name="Q",
            shape=(1, 2, 10, 4),
            dtype="float32",
        )

        self.assertEqual(tensor.to_dict()["shape"], [1, 2, 10, 4])

    def test_rejects_negative_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TensorSpec("bad", "Bad", (1, -1), "float32")


class NodeTest(unittest.TestCase):
    def test_uses_independent_mutable_defaults(self) -> None:
        first = Node(id="first", op="input", label="First")
        second = Node(id="second", op="output", label="Second")

        first.inputs.append("tensor_x")
        first.attrs["seed"] = 42

        self.assertEqual(second.inputs, [])
        self.assertEqual(second.attrs, {})


class GraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = Graph()
        self.x = self.graph.add_tensor(
            TensorSpec("tensor_x", "Input", (1, 3, 8), "float32")
        )
        self.q = self.graph.add_tensor(
            TensorSpec("tensor_q", "Q", (1, 2, 3, 4), "float32")
        )
        self.input_node = self.graph.add_node(
            Node("input", "input", "Input", outputs=[self.x.id])
        )
        self.q_node = self.graph.add_node(
            Node(
                "q_proj",
                "linear",
                "Q Projection",
                inputs=[self.x.id],
                outputs=[self.q.id],
            )
        )

    def test_builds_and_serializes_graph(self) -> None:
        edge = self.graph.add_edge(Edge("input", "q_proj", self.x.id))

        payload = self.graph.to_dict()

        self.assertEqual(edge.tensor_id, self.x.id)
        self.assertEqual(payload["nodes"][1]["label"], "Q Projection")
        self.assertEqual(
            payload["edges"],
            [{"source": "input", "target": "q_proj", "tensor_id": "tensor_x"}],
        )
        self.assertEqual(payload["tensors"]["tensor_q"]["shape"], [1, 2, 3, 4])

    def test_rejects_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "Tensor id already exists"):
            self.graph.add_tensor(self.x)

        with self.assertRaisesRegex(ValueError, "Node id already exists"):
            self.graph.add_node(
                Node("q_proj", "linear", "Duplicate", inputs=[self.x.id])
            )

    def test_rejects_unknown_tensor_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown tensors"):
            self.graph.add_node(
                Node("bad", "linear", "Bad Node", inputs=["missing"])
            )

    def test_rejects_edge_with_invalid_data_flow(self) -> None:
        with self.assertRaisesRegex(ValueError, "not produced"):
            self.graph.add_edge(Edge("q_proj", "input", self.x.id))


if __name__ == "__main__":
    unittest.main()
