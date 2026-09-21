"""Central configuration.

Every tunable lives here so no module reads os.environ directly.
Import the singleton: `from config.settings import settings`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional: without it we just read real env vars
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _secret(key: str) -> str:
    """Read a value from Streamlit's secrets store, if one is present.

    On Streamlit Community Cloud there is no `.env` - it is gitignored and
    never deployed. Keys are pasted into the app's Secrets panel instead,
    which surfaces them through `st.secrets`. This lookup is guarded so the
    CLI and the FastAPI service, which never import Streamlit, are
    unaffected.
    """
    try:
        import streamlit as st

        return str(st.secrets[key]).strip()
    except Exception:
        # No Streamlit, no secrets file, or no such key - all mean
        # "fall back to the environment".
        return ""


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key, "").strip()
    return value or _secret(key) or default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # ---- paths ----
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    cache_dir: Path = ROOT / "data" / "cache"
    kb_dir: Path = ROOT / "data" / "knowledge_base"
    vectorstore_dir: Path = ROOT / "data" / "vectorstore"

    # ---- LLM ----
    # Which provider drives planning and answer generation.
    #   groq      - free tier, open-weight models, nothing to download
    #   anthropic - Claude (paid)
    #   none      - offline: rule-based planning + template answers
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "groq"))

    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "openai/gpt-oss-120b")
    )

    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-5"))
    effort: str = field(default_factory=lambda: _env("ANTHROPIC_EFFORT", "high"))
    max_tokens: int = 4096
    # Server-side refusal fallback: if the primary model declines a request,
    # the API retries it on a fallback model inside the same call.
    enable_refusal_fallback: bool = True
    # Higher temperature for conversation, near-zero for structured planning:
    # a plan should be reproducible, an explanation should not read like a form.
    chat_temperature: float = 0.6
    plan_temperature: float = 0.1
    # Free tiers cap tokens-per-minute, so bursts return 429 with a short
    # Retry-After. The SDK honours that header; it just needs more attempts
    # than its default of 2 to ride out a sustained burst.
    llm_max_retries: int = 6

    # ---- weather providers ----
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_geocode_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    openweather_api_key: str = field(default_factory=lambda: _env("OPENWEATHER_API_KEY"))
    request_timeout: int = 30
    cache_ttl_seconds: int = field(default_factory=lambda: _env_int("CACHE_TTL_SECONDS", 1800))

    # ---- RAG ----
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "hash"))
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "simple"))
    # Hashed-vector width. Must comfortably exceed the vocabulary size or
    # unrelated words collide into the same dimension and retrieval degrades.
    embedding_dim: int = 8192
    chunk_size: int = 700             # characters
    chunk_overlap: int = 120
    top_k: int = 4

    # ---- analysis ----
    anomaly_z_threshold: float = 2.0
    baseline_years: int = 10          # years of history used as the "normal" baseline
    default_forecast_days: int = 7

    # ---- misc ----
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    @property
    def llm_enabled(self) -> bool:
        """False -> the system runs in offline mode (no API calls)."""
        backend = self.llm_backend.lower()
        if backend == "groq":
            return bool(self.groq_api_key)
        if backend == "anthropic":
            return bool(self.anthropic_api_key)
        return False

    @property
    def llm_label(self) -> str:
        """What to show the user in the UI."""
        if not self.llm_enabled:
            return "offline"
        if self.llm_backend.lower() == "groq":
            return self.groq_model
        return self.model

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.cache_dir, self.kb_dir, self.vectorstore_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
