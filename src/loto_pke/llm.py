"""Local-only LLM interfaces and strict JSON response handling."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from loto_pke.common import require_module


class TextGenerator(Protocol):
    def generate(self, system: str, prompt: str, images: list[Path] | None = None) -> str:
        """Return the raw assistant response (not a silently repaired object)."""


@dataclass
class OllamaGenerator:
    """Qwen3:14B or another local Ollama checkpoint, without external API calls."""

    model: str = "qwen3:14b"
    endpoint: str = "http://127.0.0.1:11434"
    seed: int = 42
    temperature: float = 0.0
    max_new_tokens: int = 1024
    timeout_seconds: int = 600

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Only loopback Ollama endpoints are permitted; use SSH tunneling for a remote service")

    def generate(self, system: str, prompt: str, images: list[Path] | None = None) -> str:
        user: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            image_module = require_module("PIL.Image", "Install Pillow for Ollama image input")
            encoded = []
            for image_path in images:
                with image_module.open(image_path) as image:
                    with io.BytesIO() as stream:
                        image.convert("RGB").save(stream, format="PNG")
                        encoded.append(base64.b64encode(stream.getvalue()).decode("ascii"))
            user["images"] = encoded

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, user],
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "seed": self.seed,
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        }
        request = Request(
            self.endpoint.rstrip("/") + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.load(response)
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Local Ollama request failed: {exc}") from exc
        content = result.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response did not contain message.content")
        return content


def parse_json_response(text: str) -> dict[str, Any]:
    """Accept strict JSON or a single fenced JSON block; never invent fields."""

    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json"):]
    elif stripped.startswith("```"):
        stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    value = json.loads(stripped.strip())
    if not isinstance(value, dict):
        raise ValueError("Expected the model to return a JSON object")
    return value
