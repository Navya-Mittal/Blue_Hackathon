"""LLM client with three modes and a full audit trail.

offline : no LLM. A deterministic stand-in (supplied by the caller) produces the step's output.
replay  : responses are read from llm_cache/ (recorded earlier in live mode). Misses raise ReplayMiss.
live    : a real provider is called; the response is cached and audited.

The cache key is sha256(task, system, user) so prompts must be deterministic (no timestamps).
The live providers below are thin adapters. They have NOT been exercised against a real API in
this repository's test suite (no API key in CI); unit tests use FakeProvider instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

from .audit import Audit


class ReplayMiss(RuntimeError):
    """Raised in replay mode when no recorded response exists for a prompt."""


class Provider(Protocol):
    """What a live LLM adapter must offer: a name and a `complete` method."""

    name: str

    def complete(
        self, system: str, user: str, *, temperature: float, max_tokens: int
    ) -> tuple[str, dict]:
        """Send one request and return (reply text, token usage)."""
        ...


class AnthropicProvider:
    """Thin adapter for the Anthropic API. Not yet run against the real API."""

    def __init__(self, model: str):
        import anthropic  # lazy import so offline mode needs no SDK

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, system, user, *, temperature, max_tokens):
        """Send one request and return (reply text, token usage)."""
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        return text, usage


class OpenAIProvider:
    """Thin adapter for the OpenAI API. Not yet run against the real API."""

    def __init__(self, model: str):
        from openai import OpenAI  # lazy import

        self.client = OpenAI()  # reads OPENAI_API_KEY (and OPENAI_BASE_URL if set)
        self.model = model
        self.name = f"openai:{model}"

    def complete(self, system, user, *, temperature, max_tokens):
        """Send one request and return (reply text, token usage)."""
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = {
            "input_tokens": getattr(r.usage, "prompt_tokens", None),
            "output_tokens": getattr(r.usage, "completion_tokens", None),
        }
        return r.choices[0].message.content or "", usage


def make_provider(cfg_llm: dict) -> Provider:
    """Create the live provider named in the config (anthropic or openai)."""
    model = os.environ.get("BLUE_LLM_MODEL") or cfg_llm.get("model") or ""
    if not model:
        raise ValueError(
            "Live mode needs a model name: set llm.model in config.yaml or BLUE_LLM_MODEL."
        )
    kind = cfg_llm.get("provider", "anthropic")
    if kind == "anthropic":
        return AnthropicProvider(model)
    if kind == "openai":
        return OpenAIProvider(model)
    raise ValueError(f"Unknown provider: {kind}")


@dataclass
class LLMResult:
    """The text from one LLM step and where it came from."""

    text: str
    source: str  # offline-standin | cache | live:<provider:model>


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class LLMClient:
    """Runs an LLM step in offline, replay or live mode and logs every call."""

    def __init__(
        self,
        mode: str,
        cache_dir: Path,
        audit: Audit,
        provider: Optional[Provider] = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
    ):
        if mode not in ("offline", "replay", "live"):
            raise ValueError(f"mode must be offline|replay|live, got {mode!r}")
        if mode == "live" and provider is None:
            raise ValueError("live mode requires a provider")
        self.mode = mode
        self.cache_dir = Path(cache_dir)
        self.audit = audit
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def complete(
        self, task: str, system: str, user: str, offline: Optional[Callable[[], str]] = None
    ) -> LLMResult:
        """Return the text for one task. Every call is written to the audit log."""
        key = _sha(task, system, user)
        t0 = time.perf_counter()

        if self.mode == "offline":
            if offline is None:
                raise RuntimeError(
                    f"Task {task!r} has no offline stand-in; use replay or live mode."
                )
            text, source, usage = offline(), "offline-standin", {}
        elif self.mode == "replay":
            p = self._cache_path(key)
            if not p.exists():
                raise ReplayMiss(
                    f"No cached response for task {task!r} (key {key[:12]}). "
                    f"Record it once with --mode live, then commit llm_cache/."
                )
            rec = json.loads(p.read_text(encoding="utf-8"))
            text, source, usage = rec["response"], "cache", rec.get("usage", {})
        else:
            assert self.provider is not None
            text, usage = self.provider.complete(
                system,
                user,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            source = f"live:{self.provider.name}"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(key).write_text(
                json.dumps(
                    {
                        "task": task,
                        "provider": self.provider.name,
                        "usage": usage,
                        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "response": text,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )

        self.audit.event(
            "llm_call",
            task=task,
            source=source,
            mode=self.mode,
            key=key[:16],
            prompt_sha=_sha(system, user)[:16],
            response_sha=_sha(text)[:16],
            seconds=round(time.perf_counter() - t0, 3),
            usage=usage,
        )
        return LLMResult(text=text, source=source)
