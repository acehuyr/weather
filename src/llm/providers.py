"""LLM providers behind one interface.

The rest of the codebase never imports a vendor SDK. It asks for a provider
and gets something with `complete` and `chat` - so swapping Groq for Claude,
or adding a local Ollama backend later, touches this file and nothing else.

Providers implemented:

* **GroqProvider** - free tier, open-weight models served on Groq's
  hardware. Note that Groq retires model IDs without notice; a 404 means
  a stale `GROQ_MODEL`, not a bad key (`scripts/list_models.py` lists
  what a key can use). Nothing to download, works on any machine, deploys to free
  hosting. This is the default.
* **AnthropicProvider** - Claude. Higher quality, but paid.
* **NullProvider** - no credentials: the system falls back to rule-based
  planning and template answers rather than failing.
"""
from __future__ import annotations

from typing import Protocol

from config.settings import settings
from src.core.errors import LLMError
from src.core.logging import get_logger

log = get_logger("llm.providers")


class Provider(Protocol):
    name: str
    model: str

    @property
    def available(self) -> bool: ...

    def chat(self, messages: list, temperature: float = 0.6,
             max_tokens: int = None, json_mode: bool = False) -> str: ...


# --------------------------------------------------------------------------
class NullProvider:
    """No credentials configured. Everything degrades to offline behaviour."""

    name = "none"
    model = "offline"

    @property
    def available(self) -> bool:
        return False

    def chat(self, messages: list, temperature: float = 0.6,
             max_tokens: int = None, json_mode: bool = False) -> str:
        raise LLMError("No LLM is configured")


# --------------------------------------------------------------------------
class GroqProvider:
    """Groq free tier. OpenAI-compatible chat completions."""

    name = "groq"

    def __init__(self) -> None:
        self.model = settings.groq_model
        self._client = None
        if not settings.groq_api_key:
            log.warning("GROQ_API_KEY not set - running offline")
            return
        try:
            from groq import Groq

            # The free tier caps tokens-per-minute (8k at the time of
            # writing), so a burst of calls returns 429 with a short
            # Retry-After. The SDK honours that header, but two retries is
            # not enough to ride out a sustained burst - an evaluation run,
            # or a user asking several questions quickly.
            self._client = Groq(
                api_key=settings.groq_api_key,
                max_retries=settings.llm_max_retries,
            )
            log.info("LLM ready: groq/%s", self.model)
        except ImportError:
            log.error("groq package missing - run: pip install groq")

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(self, messages: list, temperature: float = 0.6,
             max_tokens: int = None, json_mode: bool = False) -> str:
        if not self.available:
            raise LLMError("Groq is not configured")

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or settings.max_tokens,
        }
        if json_mode:
            # Constrains output to syntactically valid JSON. The schema itself
            # still has to be described in the prompt.
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        usage = getattr(response, "usage", None)
        if usage:
            log.info("tokens in=%s out=%s",
                     getattr(usage, "prompt_tokens", "?"),
                     getattr(usage, "completion_tokens", "?"))

        text = response.choices[0].message.content
        if not text or not text.strip():
            raise LLMError("Model returned no content")
        return text


# --------------------------------------------------------------------------
class AnthropicProvider:
    """Claude. Kept so the project is not locked to one vendor."""

    name = "anthropic"

    def __init__(self) -> None:
        self.model = settings.model
        self._client = None
        if not settings.anthropic_api_key:
            log.warning("ANTHROPIC_API_KEY not set - running offline")
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            log.info("LLM ready: %s", self.model)
        except ImportError:
            log.error("anthropic package missing - run: pip install anthropic")

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(self, messages: list, temperature: float = 0.6,
             max_tokens: int = None, json_mode: bool = False) -> str:
        if not self.available:
            raise LLMError("Anthropic is not configured")

        # Claude takes the system prompt as its own parameter, not as a
        # message, so split it out of the OpenAI-style list.
        system = "\n\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        turns = [m for m in messages if m["role"] != "system"]

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens or settings.max_tokens,
                # A frozen system prompt caches across requests, cutting the
                # repeated input cost on the cached prefix.
                system=[{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=turns,
                thinking={"type": "adaptive"},
                output_config={"effort": settings.effort},
            )
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            raise LLMError(
                f"Request declined ({getattr(detail, 'category', 'unknown')})"
            )

        # content is a list of blocks; thinking blocks carry no .text.
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text.strip():
            raise LLMError("Model returned no text content")
        return text


# --------------------------------------------------------------------------
def build_provider() -> Provider:
    backend = settings.llm_backend.lower()
    if backend == "groq":
        provider = GroqProvider()
    elif backend == "anthropic":
        provider = AnthropicProvider()
    else:
        return NullProvider()

    return provider if provider.available else NullProvider()
