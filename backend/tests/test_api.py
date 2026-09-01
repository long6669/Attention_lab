import unittest
from unittest.mock import patch

from app.api import attention as attention_api
from app.api.attention import reset_sessions
from app.main import app
from fastapi.testclient import TestClient


class AttentionApiTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_sessions()
        self.client = TestClient(app)

    def test_run_returns_complete_attention_payload(self) -> None:
        response = self.client.post(
            "/api/run",
            json={"text": "I love learning attention"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["tokens"],
            ["I", "love", "learning", "attention"],
        )
        self.assertEqual(payload["warnings"], [])
        self.assertEqual(payload["phase"], "prefill")
        self.assertTrue(payload["session_id"])
        self.assertEqual(payload["graph"]["nodes"][0]["id"], "input")
        self.assertEqual(payload["trace"][-1]["node_id"], "output")
        self.assertEqual(
            payload["tensors"]["attention_probs"]["shape"],
            [1, 2, 4, 4],
        )
        self.assertEqual(payload["memory"]["total_bytes"], 256)
        self.assertGreater(payload["metrics"]["estimated_flops"], 0)
        self.assertEqual(payload["metrics"]["flops_basis"], "shape_estimate")
        self.assertEqual(payload["metrics"]["memory_growth"], "linear")
        self.assertEqual(payload["metrics"]["memory_bytes_per_token"], 64)
        self.assertEqual(
            payload["metrics"]["graph_nodes"],
            len(payload["graph"]["nodes"]),
        )

    def test_run_rejects_empty_input(self) -> None:
        response = self.client.post("/api/run", json={"text": "  "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Please enter some tokens.")

    def test_run_truncates_input_to_ten_tokens(self) -> None:
        text = " ".join(f"token_{index}" for index in range(12))

        response = self.client.post("/api/run", json={"text": text})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["tokens"]), 10)
        self.assertEqual(
            payload["warnings"],
            ["MVP supports up to 10 tokens. Input was truncated."],
        )

    def test_run_supports_v02_to_v04_architectures(self) -> None:
        expectations = {
            "mqa": ("kv", [1, 1, 3, 4]),
            "gqa": ("kv", [1, 2, 3, 2]),
            "rope": ("kv", [1, 2, 3, 4]),
            "mla": ("latent", [1, 3, 4]),
        }

        for architecture, (cache_kind, shape) in expectations.items():
            with self.subTest(architecture=architecture):
                response = self.client.post(
                    "/api/run",
                    json={
                        "text": "one two three",
                        "architecture": architecture,
                    },
                )
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(
                    payload["config"]["architecture"],
                    architecture,
                )
                self.assertEqual(payload["memory"]["cache_kind"], cache_kind)
                cache_key = "latent_cache" if cache_kind == "latent" else "k_cache"
                self.assertEqual(payload["memory"][cache_key]["shape"], shape)

    def test_run_rejects_unknown_architecture(self) -> None:
        response = self.client.post(
            "/api/run",
            json={"text": "one two", "architecture": "unknown"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "Unsupported attention architecture: unknown",
        )

    def test_run_supports_v05_and_v06_architectures(self) -> None:
        expectations = {
            "kda": ("recurrent", ["state_init", "decay", "erase", "write", "scan"]),
            "csa": (
                "kv",
                ["sequence_compression", "indexer", "topk", "routing"],
            ),
            "hca": (
                "kv",
                ["sequence_compression", "indexer", "topk", "routing"],
            ),
        }

        for architecture, (cache_kind, expected_ops) in expectations.items():
            with self.subTest(architecture=architecture):
                response = self.client.post(
                    "/api/run",
                    json={
                        "text": "one two three",
                        "architecture": architecture,
                    },
                )
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                ops = {node["op"] for node in payload["graph"]["nodes"]}
                self.assertEqual(payload["memory"]["cache_kind"], cache_kind)
                self.assertTrue(set(expected_ops).issubset(ops))

    def test_kda_decode_keeps_fixed_recurrent_memory(self) -> None:
        prefill = self.client.post(
            "/api/run",
            json={"text": "one two three", "architecture": "kda"},
        ).json()

        decode = self.client.post(
            "/api/decode",
            json={"session_id": prefill["session_id"]},
        )

        self.assertEqual(decode.status_code, 200)
        payload = decode.json()
        self.assertEqual(payload["memory"]["total_bytes"], 128)
        self.assertEqual(
            payload["memory"]["recurrent_state"]["shape"],
            [1, 2, 4, 4],
        )
        self.assertEqual(
            payload["cache_activity"]["update_kind"],
            "state_update",
        )
        self.assertEqual(payload["metrics"]["memory_growth"], "constant")
        self.assertEqual(payload["metrics"]["memory_bytes_per_token"], 0)

    def test_kda_state_supports_head_slice(self) -> None:
        prefill = self.client.post(
            "/api/run",
            json={"text": "one two three", "architecture": "kda"},
        ).json()

        response = self.client.post(
            "/api/memory/slice",
            json={
                "session_id": prefill["session_id"],
                "memory_id": "recurrent_state",
                "start": 0,
                "end": 1,
                "head": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["shape"], [1, 1, 4, 4])
        self.assertEqual(
            payload["axes"],
            ["batch", "head", "key_feature", "value_feature"],
        )

    def test_decode_adds_token_and_grows_kv_cache(self) -> None:
        prefill = self.client.post(
            "/api/run",
            json={"text": "I love attention"},
        ).json()

        response = self.client.post(
            "/api/decode",
            json={"session_id": prefill["session_id"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["tokens"],
            ["I", "love", "attention", "token_4"],
        )
        self.assertEqual(payload["decoded_token"], "token_4")
        self.assertEqual(payload["phase"], "decode")
        self.assertEqual(payload["cache_activity"]["read_tokens"], 3)
        self.assertEqual(payload["cache_activity"]["appended_tokens"], 1)
        self.assertEqual(payload["memory"]["k_cache"]["shape"], [1, 2, 4, 4])
        self.assertEqual(payload["memory"]["total_elements"], 64)
        self.assertEqual(payload["memory"]["total_bytes"], 256)

    def test_decode_stops_after_eleven_total_tokens(self) -> None:
        prefill = self.client.post(
            "/api/run",
            json={"text": " ".join(f"token_{index}" for index in range(10))},
        ).json()
        first_decode = self.client.post(
            "/api/decode",
            json={"session_id": prefill["session_id"]},
        )
        self.assertEqual(first_decode.status_code, 200)

        response = self.client.post(
            "/api/decode",
            json={"session_id": prefill["session_id"]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "MVP decode supports up to 11 total tokens.",
        )

    def test_decode_grows_ten_token_cache_to_eleven(self) -> None:
        tokens = [f"token_{index}" for index in range(1, 11)]
        prefill = self.client.post(
            "/api/run",
            json={"text": " ".join(tokens)},
        ).json()

        response = self.client.post(
            "/api/decode",
            json={"session_id": prefill["session_id"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["tokens"][-1], "token_11")
        self.assertEqual(payload["memory"]["tokens"], 11)
        self.assertEqual(payload["memory"]["total_bytes"], 704)

    def test_decode_rejects_unknown_session(self) -> None:
        response = self.client.post(
            "/api/decode",
            json={"session_id": "missing"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"],
            "Attention session was not found. Run prefill again.",
        )

    def test_expired_session_is_removed_on_access(self) -> None:
        prefill = self.client.post(
            "/api/run",
            json={"text": "one two three"},
        ).json()
        session = attention_api._sessions[prefill["session_id"]]
        session.last_access -= attention_api.SESSION_TTL_SECONDS

        response = self.client.post(
            "/api/decode",
            json={"session_id": prefill["session_id"]},
        )

        self.assertEqual(response.status_code, 404)
        self.assertNotIn(prefill["session_id"], attention_api._sessions)

    def test_session_capacity_evicts_least_recently_used(self) -> None:
        with patch.object(attention_api, "MAX_SESSIONS", 2):
            first = self.client.post(
                "/api/run",
                json={"text": "first"},
            ).json()
            second = self.client.post(
                "/api/run",
                json={"text": "second"},
            ).json()
            self.client.post(
                "/api/memory/slice",
                json={
                    "session_id": first["session_id"],
                    "memory_id": "k_cache",
                    "start": 0,
                    "end": 1,
                },
            )
            third = self.client.post(
                "/api/run",
                json={"text": "third"},
            ).json()

        self.assertIn(first["session_id"], attention_api._sessions)
        self.assertNotIn(second["session_id"], attention_api._sessions)
        self.assertIn(third["session_id"], attention_api._sessions)

    def test_decode_session_persists_cache_across_requests(self) -> None:
        prefill = self.client.post(
            "/api/run",
            json={"text": "one two three"},
        ).json()

        first = self.client.post(
            "/api/decode",
            json={"session_id": prefill["session_id"]},
        ).json()
        second = self.client.post(
            "/api/decode",
            json={"session_id": prefill["session_id"]},
        ).json()

        self.assertEqual(first["tokens"][-1], "token_4")
        self.assertEqual(second["tokens"][-1], "token_5")
        self.assertEqual(second["cache_activity"]["read_tokens"], 4)
        self.assertEqual(second["memory"]["k_cache"]["shape"], [1, 2, 5, 4])
        self.assertEqual(
            second["tensors"]["q_new_heads"]["shape"],
            [1, 2, 1, 4],
        )

    def test_mla_session_persists_latent_cache(self) -> None:
        prefill = self.client.post(
            "/api/run",
            json={"text": "one two three", "architecture": "mla"},
        ).json()

        decode = self.client.post(
            "/api/decode",
            json={"session_id": prefill["session_id"]},
        )

        self.assertEqual(decode.status_code, 200)
        payload = decode.json()
        self.assertEqual(payload["memory"]["latent_cache"]["shape"], [1, 4, 4])
        self.assertEqual(payload["memory"]["total_bytes"], 64)
        self.assertEqual(
            payload["tensors"]["raw_scores"]["shape"],
            [1, 2, 1, 4],
        )

    def test_memory_slice_reads_only_requested_cache_range(self) -> None:
        prefill = self.client.post(
            "/api/run",
            json={"text": "one two three four", "architecture": "mha"},
        ).json()

        response = self.client.post(
            "/api/memory/slice",
            json={
                "session_id": prefill["session_id"],
                "memory_id": "k_cache",
                "start": 1,
                "end": 3,
                "head": 0,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["shape"], [1, 1, 2, 4])
        self.assertEqual(payload["numel"], 8)
        self.assertEqual(payload["selection"]["start"], 1)
        self.assertEqual(payload["selection"]["end"], 3)

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
