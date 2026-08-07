"""The weather tool itself.

Deliberately has zero knowledge of the LLM/agent layer: it's a plain function
that takes a city name and returns real weather data, so it can be tested and
reused on its own. Uses Open-Meteo, which needs no API key -- geocoding turns
a free-text city name into coordinates, then the forecast endpoint returns
current conditions for those coordinates.
"""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10

# WMO weather codes -> human-readable condition, per Open-Meteo's `weathercode` docs.
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class WeatherLookupError(Exception):
    """Raised when a city can't be resolved or the weather API call fails."""


def get_weather(city: str) -> dict:
    """Look up current weather conditions for a city.

    Args:
        city: A place name, e.g. "Hyderabad" or "Paris, France".

    Returns:
        A dict with city, country, temperature_c, windspeed_kmh, condition.

    Raises:
        WeatherLookupError: if the city can't be geocoded or the forecast
            API doesn't return usable data.
    """
    geo_response = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 1},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    geo_response.raise_for_status()
    results = geo_response.json().get("results")
    if not results:
        raise WeatherLookupError(f"Could not find a location matching '{city}'.")
    place = results[0]

    forecast_response = requests.get(
        FORECAST_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current_weather": True,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    forecast_response.raise_for_status()
    current = forecast_response.json().get("current_weather")
    if not current:
        raise WeatherLookupError(f"Weather data unavailable for '{city}'.")

    return {
        "city": place["name"],
        "country": place.get("country", ""),
        "temperature_c": current["temperature"],
        "windspeed_kmh": current["windspeed"],
        "condition": WEATHER_CODES.get(current["weathercode"], "unknown"),
    }
