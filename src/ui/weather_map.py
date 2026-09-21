"""Interactive weather map.

Built on Folium (Leaflet) with OpenStreetMap tiles - both open source, both
free, no API key and no usage tier. Weather for each marker comes from the
same keyless Open-Meteo endpoint the rest of the project uses.

The map shows the chosen place plus the nearest towns around it, each marker
coloured by temperature and carrying a small popup of current conditions, so
you can see at a glance where it is hotter, wetter or windier nearby.
"""
from __future__ import annotations

import math

import folium

from config.settings import settings
from src.core.errors import ProviderError
from src.core.logging import get_logger
from src.core.models import Location
from src.ingestion.cache import cached_get
from src.ingestion.open_meteo import WEATHER_CODES

log = get_logger("ui.map")

# Temperature bands (deg C) -> marker colour. Folium only accepts these names.
TEMPERATURE_BANDS = [
    (40, "darkred", "Extreme heat"),
    (35, "red", "Very hot"),
    (30, "orange", "Hot"),
    (25, "beige", "Warm"),
    (20, "green", "Mild"),
    (15, "lightblue", "Cool"),
    (5, "blue", "Cold"),
    (-100, "darkblue", "Very cold"),
]


def temperature_style(celsius) -> tuple:
    """(colour, label) for a temperature, for markers and the legend."""
    if celsius is None:
        return "gray", "No data"
    for threshold, colour, label in TEMPERATURE_BANDS:
        if celsius >= threshold:
            return colour, label
    return "gray", "No data"


# Bundled so real place names appear without a paid radius-search API.
# Open-Meteo's geocoder only matches by name, never by proximity.
CITIES = [
    ("Mumbai", 19.0760, 72.8777), ("Thane", 19.2183, 72.9781),
    ("Navi Mumbai", 19.0330, 73.0297), ("Kalyan", 19.2437, 73.1355),
    ("Vasai", 19.4259, 72.8225), ("Panvel", 18.9894, 73.1175),
    ("Alibag", 18.6414, 72.8722), ("Karjat", 18.9107, 73.3233),
    ("Lonavala", 18.7546, 73.4062), ("Pune", 18.5204, 73.8567),
    ("Pimpri", 18.6298, 73.7997), ("Satara", 17.6805, 74.0183),
    ("Nashik", 19.9975, 73.7898), ("Ahmednagar", 19.0948, 74.7480),
    ("Kolhapur", 16.7050, 74.2433), ("Solapur", 17.6599, 75.9064),
    ("Aurangabad", 19.8762, 75.3433), ("Nagpur", 21.1458, 79.0882),
    ("Amravati", 20.9320, 77.7523), ("Akola", 20.7002, 77.0082),
    ("Delhi", 28.6139, 77.2090), ("Gurugram", 28.4595, 77.0266),
    ("Noida", 28.5355, 77.3910), ("Faridabad", 28.4089, 77.3178),
    ("Ghaziabad", 28.6692, 77.4538), ("Meerut", 28.9845, 77.7064),
    ("Rohtak", 28.8955, 76.6066), ("Panipat", 29.3909, 76.9635),
    ("Agra", 27.1767, 78.0081), ("Jaipur", 26.9124, 75.7873),
    ("Lucknow", 26.8467, 80.9462), ("Kanpur", 26.4499, 80.3319),
    ("Varanasi", 25.3176, 82.9739), ("Patna", 25.5941, 85.1376),
    ("Kolkata", 22.5726, 88.3639), ("Howrah", 22.5958, 88.2636),
    ("Durgapur", 23.5204, 87.3119), ("Siliguri", 26.7271, 88.3953),
    ("Bengaluru", 12.9716, 77.5946), ("Mysuru", 12.2958, 76.6394),
    ("Tumakuru", 13.3409, 77.1017), ("Hosur", 12.7409, 77.8253),
    ("Mangaluru", 12.9141, 74.8560), ("Hubballi", 15.3647, 75.1240),
    ("Chennai", 13.0827, 80.2707), ("Vellore", 12.9165, 79.1325),
    ("Puducherry", 11.9416, 79.8083), ("Coimbatore", 11.0168, 76.9558),
    ("Madurai", 9.9252, 78.1198), ("Tiruchirappalli", 10.7905, 78.7047),
    ("Hyderabad", 17.3850, 78.4867), ("Warangal", 17.9689, 79.5941),
    ("Vijayawada", 16.5062, 80.6480), ("Visakhapatnam", 17.6868, 83.2185),
    ("Ahmedabad", 23.0225, 72.5714), ("Gandhinagar", 23.2156, 72.6369),
    ("Vadodara", 22.3072, 73.1812), ("Surat", 21.1702, 72.8311),
    ("Rajkot", 22.3039, 70.8022), ("Bhopal", 23.2599, 77.4126),
    ("Indore", 22.7196, 75.8577), ("Jabalpur", 23.1815, 79.9864),
    ("Raipur", 21.2514, 81.6296), ("Ranchi", 23.3441, 85.3096),
    ("Bhubaneswar", 20.2961, 85.8245), ("Guwahati", 26.1445, 91.7362),
    ("Shimla", 31.1048, 77.1734), ("Chandigarh", 30.7333, 76.7794),
    ("Amritsar", 31.6340, 74.8723), ("Ludhiana", 30.9010, 75.8573),
    ("Dehradun", 30.3165, 78.0322), ("Jodhpur", 26.2389, 73.0243),
    ("Udaipur", 24.5854, 73.7125), ("Kochi", 9.9312, 76.2673),
    ("Thiruvananthapuram", 8.5241, 76.9366), ("Kozhikode", 11.2588, 75.7804),
    ("Panaji", 15.4909, 73.8278), ("Srinagar", 34.0837, 74.7973),
    ("Jammu", 32.7266, 74.8570), ("Cherrapunji", 25.2702, 91.7323),
]


def _km_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometres."""
    from math import asin, cos, radians, sin, sqrt

    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = (sin(d_lat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2)
    return 6371.0 * 2 * asin(sqrt(a))


def nearby_places(loc: Location, count: int = 8, radius_km: float = 160) -> list:
    """Real towns near a point, topped up with sample points if sparse.

    Named places are preferred because "Thane 29 C" means more to a reader
    than "49 km north". Where the bundled table has no neighbours - a remote
    area, or anywhere outside India - evenly spaced sample points still give
    the map useful spatial coverage.
    """
    scored = []
    for name, lat, lon in CITIES:
        km = _km_between(loc.latitude, loc.longitude, lat, lon)
        if 8 < km <= radius_km:          # skip the centre itself
            scored.append((km, Location(name=name, latitude=lat, longitude=lon)))

    scored.sort(key=lambda pair: pair[0])
    found = [place for _, place in scored[:count]]

    if len(found) < count:
        found.extend(_grid_around(loc, count - len(found)))
    return found[:count]


def _grid_around(loc: Location, count: int) -> list:
    """Evenly spaced sample points, for areas with no named neighbours."""
    offsets = [
        (0.45, 0, "N"), (-0.45, 0, "S"), (0, 0.45, "E"), (0, -0.45, "W"),
        (0.35, 0.35, "NE"), (-0.35, 0.35, "SE"),
        (0.35, -0.35, "NW"), (-0.35, -0.35, "SW"),
    ]
    points = []
    for d_lat, d_lon, bearing in offsets[:count]:
        km = _km_between(loc.latitude, loc.longitude,
                         loc.latitude + d_lat, loc.longitude + d_lon)
        points.append(Location(
            name=f"{bearing} {int(round(km))} km",
            latitude=round(loc.latitude + d_lat, 4),
            longitude=round(loc.longitude + d_lon, 4),
        ))
    return points


def fetch_points(locations: list) -> list:
    """Current conditions for many points in one request.

    Open-Meteo accepts comma-separated coordinates and returns a list, so a
    nine-marker map costs exactly one HTTP call rather than nine.
    """
    if not locations:
        return []

    try:
        payload = cached_get(
            settings.open_meteo_forecast_url,
            {
                "latitude": ",".join(str(l.latitude) for l in locations),
                "longitude": ",".join(str(l.longitude) for l in locations),
                "current": "temperature_2m,relative_humidity_2m,"
                           "apparent_temperature,precipitation,weather_code,"
                           "wind_speed_10m",
                "timezone": "auto",
            },
            ttl=900,
        )
    except ProviderError as exc:
        log.warning("bulk weather fetch failed: %s", exc)
        return [(loc, {}) for loc in locations]

    # One location comes back as a dict, several as a list.
    blocks = payload if isinstance(payload, list) else [payload]
    out = []
    for loc, block in zip(locations, blocks):
        current = dict(block.get("current", {}))
        current["condition"] = WEATHER_CODES.get(
            current.get("weather_code"), "Unknown"
        )
        out.append((loc, current))
    return out


def _popup(loc: Location, current: dict) -> str:
    if not current:
        return f"<b>{loc.name}</b><br>No data available"

    def row(label: str, value, unit: str = "") -> str:
        if value is None:
            return ""
        return (f"<tr><td style='padding-right:10px;color:#666'>{label}</td>"
                f"<td><b>{value}{unit}</b></td></tr>")

    return (
        f"<div style='font-family:system-ui;font-size:13px;min-width:180px'>"
        f"<div style='font-size:15px;font-weight:600;margin-bottom:4px'>"
        f"{loc.name}</div>"
        f"<div style='color:#666;margin-bottom:6px'>{current.get('condition','')}</div>"
        f"<table>"
        f"{row('Temperature', current.get('temperature_2m'), ' &deg;C')}"
        f"{row('Feels like', current.get('apparent_temperature'), ' &deg;C')}"
        f"{row('Humidity', current.get('relative_humidity_2m'), '%')}"
        f"{row('Rain', current.get('precipitation'), ' mm')}"
        f"{row('Wind', current.get('wind_speed_10m'), ' km/h')}"
        f"</table></div>"
    )


def build_map(centre: Location, points: list, zoom: int = 8) -> folium.Map:
    """A Folium map with one marker per point, coloured by temperature."""
    fmap = folium.Map(
        location=[centre.latitude, centre.longitude],
        zoom_start=zoom,
        # Plain OpenStreetMap: genuinely free and keyless. CartoDB's tiles
        # look nicer but now require an API key, which this project will not
        # depend on.
        tiles="OpenStreetMap",
        control_scale=True,
    )

    for loc, current in points:
        temp = current.get("temperature_2m")
        colour, _ = temperature_style(temp)
        is_centre = (abs(loc.latitude - centre.latitude) < 0.01
                     and abs(loc.longitude - centre.longitude) < 0.01)

        # A readable temperature badge sitting under each pin.
        if temp is not None:
            folium.map.Marker(
                [loc.latitude, loc.longitude],
                icon=folium.DivIcon(
                    icon_size=(64, 20), icon_anchor=(32, -6),
                    html=(
                        "<div style='font-family:system-ui;font-size:12px;"
                        "font-weight:700;text-align:center;color:#111;"
                        "background:rgba(255,255,255,.85);border-radius:9px;"
                        f"padding:1px 4px'>{temp}&deg;</div>"
                    ),
                ),
            ).add_to(fmap)

        folium.Marker(
            [loc.latitude, loc.longitude],
            popup=folium.Popup(_popup(loc, current), max_width=260),
            tooltip=f"{loc.name} - {temp} C" if temp is not None else loc.name,
            icon=folium.Icon(
                color=colour,
                icon="star" if is_centre else "cloud",
                prefix="glyphicon",
            ),
        ).add_to(fmap)

    # Rain gets its own translucent circles - it is the thing people actually
    # want to see on a map, and it does not read from a pin colour.
    for loc, current in points:
        rain = current.get("precipitation")
        if rain:
            folium.Circle(
                [loc.latitude, loc.longitude],
                radius=6000 + float(rain) * 2500,
                color="#2b7bba", fill=True, fill_opacity=0.22, weight=1,
                tooltip=f"{loc.name}: {rain} mm now",
            ).add_to(fmap)

    return fmap


def zoom_for(points: list, map_width_px: int = 700) -> int:
    """Pick a zoom level that frames every marker.

    Leaflet's own `fitBounds` is the obvious tool, but it measures the map
    container - and inside an inactive Streamlit tab that container has zero
    width, so it settles on a whole-world view. Computing the level from the
    coordinate span instead is independent of when the element is laid out.
    """
    if len(points) < 2:
        return 9

    lats = [loc.latitude for loc, _ in points]
    lons = [loc.longitude for loc, _ in points]
    span = max(max(lats) - min(lats), max(lons) - min(lons), 0.05)
    span *= 1.35                      # breathing room around the edge markers

    # 360 degrees spans 256 px at zoom 0, doubling each level.
    zoom = math.log2(360 * map_width_px / (256 * span))
    return int(max(3, min(12, math.floor(zoom))))


def legend_items() -> list:
    """(colour, label, range) rows for the UI legend."""
    rows = []
    previous = None
    for threshold, colour, label in TEMPERATURE_BANDS:
        span = (f"{threshold}+" if previous is None
                else f"{threshold} to {previous}")
        rows.append((colour, label, f"{span} C"))
        previous = threshold
    return rows[:-1]     # drop the -100 sentinel band
