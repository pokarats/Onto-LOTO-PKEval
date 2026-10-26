"""Local-only model transport tests with no network or model call."""

from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loto_pke.llm import OllamaGenerator, parse_json_response

try:
    from PIL import Image
except ImportError:
    Image = None


class TestOllamaTransport(unittest.TestCase):
    def test_remote_api_endpoint_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OllamaGenerator(endpoint="https://example.com")

    def test_json_reply(self) -> None:
        self.assertEqual(parse_json_response('```json\n{"answer":true}\n```'), {"answer": True})
        with self.assertRaises(json.JSONDecodeError):
            parse_json_response("not json")

    @unittest.skipIf(Image is None, "Pillow is optional")
    def test_image_is_explicitly_converted_to_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.webp"
            Image.new("RGB", (2, 2), "white").save(path)
            response = io.BytesIO(b'{"message":{"content":"{\\"nodes\\":[]}"}}')

            def respond(request, timeout):
                data = json.loads(request.data.decode("utf-8"))
                img_data = base64.b64decode(data["messages"][1]["images"][0])
                self.assertTrue(img_data.startswith(b"\x89PNG"))
                self.assertEqual(data["options"]["temperature"], 0.0)
                return response

            with patch("loto_pke.llm.urlopen", side_effect=respond):
                result = OllamaGenerator().generate("system", "prompt", [path])
            self.assertEqual(result, '{"nodes":[]}')


if __name__ == "__main__":
    unittest.main()
