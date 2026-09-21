"""List the Groq models this API key can actually use.

Groq retires model IDs without notice. When that happens the API returns a
404 saying the model "does not exist or you do not have access to it" - which
reads like an authentication problem but is not one. A 401 means the key is
wrong; a 404 means the model name is stale.

    python scripts/list_models.py

Then put a working ID in .env as GROQ_MODEL.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402

# Speech, moderation and embedding models cannot answer a chat request.
NOT_CHAT = ("whisper", "tts", "guard", "embed", "orpheus")


def main() -> int:
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set in .env")
        return 1

    try:
        from groq import Groq
    except ImportError:
        print("groq package missing - run: pip install groq")
        return 1

    client = Groq(api_key=settings.groq_api_key)
    try:
        models = client.models.list().data
    except Exception as exc:  # noqa: BLE001
        print(f"Could not reach Groq: {type(exc).__name__}: {exc}")
        return 1

    chat = sorted(
        (m for m in models if not any(x in m.id.lower() for x in NOT_CHAT)),
        key=lambda m: m.id,
    )

    print(f"Chat models available to this key ({len(chat)}):\n")
    for model in chat:
        marker = " <- currently configured" if model.id == settings.groq_model else ""
        context = getattr(model, "context_window", "?")
        print(f"  {model.id:34s} context={context}{marker}")

    if settings.groq_model not in {m.id for m in chat}:
        print(f"\n  WARNING: GROQ_MODEL='{settings.groq_model}' is not in this "
              "list.\n  Pick one above and set it in .env.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
