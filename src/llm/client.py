"""One LLM interface for the whole project.

Three things the rest of the code needs:

* `complete(system, user)`  - one-shot text
* `parse(system, user, Model)` - a validated Pydantic object
* `converse(system, history, user)` - multi-turn chat that remembers

`parse` is the interesting one. Providers differ in how they constrain output
(Claude has schema-enforced structured outputs, Groq has a JSON mode that only
guarantees *valid* JSON, not the right shape), so the schema is described in
the prompt, the JSON is validated against Pydantic here, and one repair retry
is attempted before giving up. Callers get a typed object or an exception -
never half-parsed prose.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from config.settings import settings
from src.core.errors import LLMError
from src.core.logging import get_logger
from src.llm.providers import build_provider

log = get_logger("llm.client")

T = TypeVar("T", bound=BaseModel)

# Models sometimes wrap JSON in markdown fences despite instructions.
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    match = FENCE.search(text)
    if match:
        return match.group(1)
    # Otherwise take the outermost braces, ignoring any preamble.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def _schema_hint(schema: Type[BaseModel]) -> str:
    """A compact description of the expected object for the prompt."""
    spec = schema.model_json_schema()
    lines = []
    for name, prop in spec.get("properties", {}).items():
        kind = prop.get("type", "")
        if "enum" in prop:
            kind = " | ".join(repr(v) for v in prop["enum"])
        elif kind == "array":
            kind = f"array of {prop.get('items', {}).get('type', 'string')}"
        note = prop.get("description", "")
        lines.append(f'  "{name}": {kind}   // {note}')
    return "{\n" + "\n".join(lines) + "\n}"


class LLMClient:
    def __init__(self) -> None:
        self.provider = build_provider()

    @property
    def available(self) -> bool:
        return self.provider.available

    @property
    def label(self) -> str:
        return f"{self.provider.name}/{self.provider.model}"

    # ------------------------------------------------------------------
    def complete(self, system: str, user: str,
                 temperature: float = None, max_tokens: int = None) -> str:
        return self.provider.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=settings.chat_temperature if temperature is None else temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    def converse(self, system: str, history: list, user: str,
                 temperature: float = None, max_tokens: int = None) -> str:
        """Multi-turn chat. `history` is [{"role": ..., "content": ...}]."""
        messages = [{"role": "system", "content": system}]
        # Keep the last few turns only - the weather context is re-attached
        # each turn anyway, and a long tail just costs tokens.
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": user})
        return self.provider.chat(
            messages,
            temperature=settings.chat_temperature if temperature is None else temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    def parse(self, system: str, user: str, schema: Type[T],
              temperature: float = None) -> T:
        """Structured output validated against a Pydantic model."""
        if not self.available:
            raise LLMError("LLM is not configured")

        instruction = (
            f"{system}\n\n"
            "Reply with a single JSON object and nothing else - no prose, no "
            "markdown fence. It must match this shape exactly:\n\n"
            f"{_schema_hint(schema)}"
        )
        temperature = settings.plan_temperature if temperature is None else temperature

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user},
        ]

        last_error = ""
        for attempt in range(2):
            raw = self.provider.chat(
                messages, temperature=temperature, json_mode=True
            )
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                log.warning("structured parse failed (attempt %d): %s",
                            attempt + 1, last_error[:200])
                # Hand the model its own output and the error, and let it fix
                # the object rather than regenerate blind.
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content":
                        f"That did not validate: {last_error[:400]}\n"
                        "Return only the corrected JSON object."},
                ]

        raise LLMError(f"Could not parse a valid {schema.__name__}: {last_error[:200]}")


_singleton: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    """Process-wide client - one construction, one connection pool."""
    global _singleton
    if _singleton is None:
        _singleton = LLMClient()
    return _singleton


def reset_llm() -> None:
    """Force a rebuild - used when settings change at runtime."""
    global _singleton
    _singleton = None
