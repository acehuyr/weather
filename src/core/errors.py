"""Typed exceptions so failures are attributable to a layer."""
from __future__ import annotations


class WeatherIQError(Exception):
    """Base class for all project errors."""


class ProviderError(WeatherIQError):
    """A weather data provider failed or returned something unusable."""


class LocationNotFound(WeatherIQError):
    """Geocoding could not resolve the place name."""


class AnalysisError(WeatherIQError):
    """Not enough data (or wrong shape) to run an analysis."""


class RetrievalError(WeatherIQError):
    """The RAG index is missing or unreadable."""


class LLMError(WeatherIQError):
    """The language model call failed."""
