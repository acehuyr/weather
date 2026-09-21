"""Single logging setup used by every module."""
from __future__ import annotations

import logging
import sys

from config.settings import settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        # Windows consoles default to cp1252, which cannot encode the
        # diacritics in real place names - "Nagaur" comes back from the
        # geocoder as "Nāgaur" and logging it raises UnicodeEncodeError.
        # Force UTF-8 so international names print instead of crashing.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # pragma: no cover - non-TTY streams
            pass

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("weatheriq")
        root.addHandler(handler)
        root.setLevel(settings.log_level.upper())
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"weatheriq.{name}")
