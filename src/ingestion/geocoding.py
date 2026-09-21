"""Place name -> coordinates.

Uses Open-Meteo's free geocoding API, with a small offline table so the demo
still works on a flaky conference wifi.
"""
from __future__ import annotations

from config.settings import settings
from src.core.errors import LocationNotFound, ProviderError
from src.core.logging import get_logger
from src.core.models import Location
from src.ingestion.cache import cached_get

log = get_logger("ingestion.geocode")

# Offline fallback - extend freely.
_FALLBACK: dict[str, Location] = {
    "mumbai": Location("Mumbai", 19.0760, 72.8777, "India", "Maharashtra"),
    "delhi": Location("Delhi", 28.6139, 77.2090, "India", "Delhi"),
    "new delhi": Location("New Delhi", 28.6139, 77.2090, "India", "Delhi"),
    "pune": Location("Pune", 18.5204, 73.8567, "India", "Maharashtra"),
    "bengaluru": Location("Bengaluru", 12.9716, 77.5946, "India", "Karnataka"),
    "bangalore": Location("Bengaluru", 12.9716, 77.5946, "India", "Karnataka"),
    "chennai": Location("Chennai", 13.0827, 80.2707, "India", "Tamil Nadu"),
    "kolkata": Location("Kolkata", 22.5726, 88.3639, "India", "West Bengal"),
    "hyderabad": Location("Hyderabad", 17.3850, 78.4867, "India", "Telangana"),
    "ahmedabad": Location("Ahmedabad", 23.0225, 72.5714, "India", "Gujarat"),
    "jaipur": Location("Jaipur", 26.9124, 75.7873, "India", "Rajasthan"),
    "nagpur": Location("Nagpur", 21.1458, 79.0882, "India", "Maharashtra"),
    "lucknow": Location("Lucknow", 26.8467, 80.9462, "India", "Uttar Pradesh"),
    "shimla": Location("Shimla", 31.1048, 77.1734, "India", "Himachal Pradesh"),
    "cherrapunji": Location("Cherrapunji", 25.2702, 91.7323, "India", "Meghalaya"),
    "london": Location("London", 51.5074, -0.1278, "United Kingdom"),
    "new york": Location("New York", 40.7128, -74.0060, "United States"),
    "tokyo": Location("Tokyo", 35.6762, 139.6503, "Japan"),
}

DEFAULT_LOCATION = _FALLBACK["mumbai"]


def detect_location_from_ip() -> Location:
    """Approximate location from the public IP address.

    Uses ip-api.com's free endpoint - no key, no signup. Accuracy is
    city-level at best and can be wrong on mobile networks or a VPN, so the
    UI always lets the user override it. Raises LocationNotFound rather than
    guessing if the lookup fails.
    """
    try:
        payload = cached_get(
            "http://ip-api.com/json/",
            {"fields": "status,country,regionName,city,lat,lon"},
            ttl=60 * 60 * 6,
        )
    except ProviderError as exc:
        raise LocationNotFound(f"Could not detect your location: {exc}") from exc

    if payload.get("status") != "success" or not payload.get("city"):
        raise LocationNotFound("IP lookup returned no usable location.")

    return Location(
        name=payload["city"],
        latitude=float(payload["lat"]),
        longitude=float(payload["lon"]),
        country=payload.get("country", ""),
        admin1=payload.get("regionName", ""),
    )


def geocode(place: str) -> Location:
    """Resolve a place name. Raises LocationNotFound if nothing matches."""
    query = (place or "").strip()
    if not query:
        raise LocationNotFound("No location was given.")

    try:
        payload = cached_get(
            settings.open_meteo_geocode_url,
            {"name": query, "count": 1, "language": "en", "format": "json"},
            ttl=60 * 60 * 24 * 30,   # place coordinates do not move
        )
        results = payload.get("results") or []
        if results:
            r = results[0]
            return Location(
                name=r["name"],
                latitude=float(r["latitude"]),
                longitude=float(r["longitude"]),
                country=r.get("country", ""),
                admin1=r.get("admin1", ""),
                timezone=r.get("timezone", "auto"),
            )
        log.info("geocoder returned no match for %r", query)
    except ProviderError as exc:
        log.warning("geocoder unavailable (%s) - trying offline table", exc)

    hit = _FALLBACK.get(query.lower())
    if hit:
        return hit
    raise LocationNotFound(f"Could not find a place called {query!r}.")
