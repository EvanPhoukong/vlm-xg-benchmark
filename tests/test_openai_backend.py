"""Offline checks for image requests, API failures, and saved benchmark outputs."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import AuthenticationError, OpenAI
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xg_vlm_pilot as pilot


def response_payload(text='{"xg": 0.2, "reason": "The keeper covers the goal."}', status="completed"):
    return {
        "id": "resp_test", "object": "response", "created_at": 0,
        "model": "gpt-6-luna", "status": status,
        "output": [] if text is None else [{
            "id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}]}


def sample_shots():
    rows = []
    for i in range(20):
        x, y = 100.0 + i % 4, 35.0 + i % 5
        distance, angle = pilot.geometry(x, y)
        rows.append({
            "match_id": i // 2, "shot_id": f"shot-{i}", "x": x, "y": y,
            "ff": [{"teammate": False, "location": [118, 40],
                    "position": {"name": "Goalkeeper"}}],
            "goal": i % 2, "sb_xg": 0.15, "body_part": "Right Foot",
            "play_pattern": "Regular Play", "distance": distance, "angle": angle,
            "header": 0, "defenders_in_cone": 1})
    return pd.DataFrame(rows)


class OpenAIBackendTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, pilot, "_openai_client", pilot._openai_client)
        pilot._openai_client = None
        env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

    def use_transport(self, handler, max_retries=0):
        client = OpenAI(api_key="test-key", max_retries=max_retries,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.close)
        pilot._openai_client = client

    def test_image_prompt_and_schema_reach_responses_api(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=response_payload())

        self.use_transport(handler)
        image = base64.b64encode(b"test-image").decode()
        prompt = pilot.PROMPT.format(context="")
        text = pilot.ask_openai(image, prompt, "gpt-6-luna")
        self.assertEqual(pilot.parse(text)[0], 0.2)
        self.assertEqual(requests[0].url.path, "/v1/responses")
        body = json.loads(requests[0].content)
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertFalse(body["store"])
        content = body["input"][0]["content"]
        self.assertEqual(content[0]["image_url"], f"data:image/png;base64,{image}")
        self.assertEqual(content[0]["detail"], "high")
        self.assertEqual(content[1]["text"], prompt)
        schema = body["text"]["format"]
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["schema"]["properties"]["xg"],
                         {"type": "number", "minimum": 0, "maximum": 1})

    def test_sdk_recovers_from_service_unavailable(self):
        attempts = []

        def handler(request):
            attempts.append(request)
            if len(attempts) == 1:
                return httpx.Response(503, headers={"retry-after": "0.001"},
                                      json={"error": {"message": "Busy", "type": "server_error"}})
            return httpx.Response(200, json=response_payload())

        self.use_transport(handler, max_retries=1)
        self.assertEqual(pilot.parse(pilot.ask_openai("aW1hZ2U=", "prompt", "gpt-6-luna"))[0], 0.2)
        self.assertEqual(len(attempts), 2)

    def test_authentication_failure_is_not_retried(self):
        attempts = []

        def handler(request):
            attempts.append(request)
            return httpx.Response(401, json={"error": {
                "message": "Invalid test credentials", "type": "invalid_request_error"}})

        self.use_transport(handler, max_retries=6)
        with self.assertRaises(AuthenticationError):
            pilot.ask_openai("aW1hZ2U=", "prompt", "gpt-6-luna")
        self.assertEqual(len(attempts), 1)

    def test_incomplete_response_is_rejected(self):
        self.use_transport(lambda request: httpx.Response(
            200, json=response_payload(text=None, status="incomplete")))
        with self.assertRaisesRegex(RuntimeError, "no complete prediction"):
            pilot.ask_openai("aW1hZ2U=", "prompt", "gpt-6-luna")

    def test_missing_key_fails_before_loading_matches(self):
        stderr = io.StringIO()
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
                patch.object(sys, "argv", ["pilot", "--backend", "openai"]), \
                patch.object(pilot, "load_shots") as load, contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                pilot.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Set OPENAI_API_KEY", stderr.getvalue())
        load.assert_not_called()

    def test_json_numbers_and_quoted_rationales_are_preserved(self):
        payload = json.dumps({"xg": 1e-7, "reason": 'A "blocked" lane.'})
        self.assertEqual(pilot.parse(payload),
                         (0.001, 'A "blocked" lane.'))

    def test_legacy_response_format_is_still_accepted(self):
        self.assertEqual(pilot.parse('```json\n{"xg": 25, "reason": "Clear chance."}\n```'),
                         (0.25, "Clear chance."))

    def test_cli_writes_images_predictions_and_reuses_cache(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=response_payload())

        self.use_transport(handler)
        with tempfile.TemporaryDirectory() as directory:
            old_cwd = os.getcwd()
            try:
                os.chdir(directory)
                with patch.object(sys, "argv", ["pilot", "--backend", "openai", "--n", "20"]), \
                        patch.object(pilot, "load_shots", return_value=sample_shots()), \
                        patch.object(pilot, "IMG_DIR", "results/images"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    pilot.main()
                    output = Path("results/gpt_6_luna")
                    predictions = pd.read_csv(output / "shots.csv")
                    self.assertEqual(len(predictions), 20)
                    self.assertTrue((predictions.vlm_xg == 0.2).all())
                    self.assertIn("baseline_xg", predictions)
                    self.assertIn("vlm_reason", predictions)
                    self.assertEqual(len(list(Path("results/images").glob("*.png"))), 20)
                    self.assertTrue((output / "calibration.png").is_file())
                    self.assertIn("openai/gpt-6-luna", (output / "summary.txt").read_text())
                    with patch.object(pilot, "ask_openai", side_effect=AssertionError("Cache missed")):
                        pilot.main()
                    self.assertEqual(len(requests), 20)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
