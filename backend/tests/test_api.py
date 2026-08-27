import unittest

from fastapi.testclient import TestClient

from app.api.attention import reset_sessions
from app.main import app


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
                cache_key = (
                    "latent_cache" if cache_kind == "latent" else "k_cache"
                )
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

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
