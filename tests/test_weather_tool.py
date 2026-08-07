"""Tests for the weather tool. All HTTP calls are faked -- no network access
or API key required to run these."""

import pytest
import requests

from weather_agent.weather_tool import WeatherLookupError, get_weather


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def test_get_weather_success(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(url)
        if "geocoding" in url:
            return FakeResponse(
                {
                    "results": [
                        {
                            "name": "Hyderabad",
                            "country": "India",
                            "latitude": 17.38,
                            "longitude": 78.48,
                        }
                    ]
                }
            )
        return FakeResponse(
            {"current_weather": {"temperature": 31.5, "windspeed": 12.0, "weathercode": 1}}
        )

    monkeypatch.setattr("weather_agent.weather_tool.requests.get", fake_get)

    result = get_weather("Hyderabad")

    assert result == {
        "city": "Hyderabad",
        "country": "India",
        "temperature_c": 31.5,
        "windspeed_kmh": 12.0,
        "condition": "mainly clear",
    }
    # Geocode first, then forecast -- two calls, in that order.
    assert len(calls) == 2
    assert "geocoding" in calls[0]
    assert "forecast" in calls[1]


def test_get_weather_unknown_city_raises(monkeypatch):
    monkeypatch.setattr(
        "weather_agent.weather_tool.requests.get",
        lambda url, params, timeout: FakeResponse({"results": []}),
    )

    with pytest.raises(WeatherLookupError):
        get_weather("Notarealplacexyz")


def test_get_weather_unknown_condition_code_falls_back(monkeypatch):
    def fake_get(url, params, timeout):
        if "geocoding" in url:
            return FakeResponse(
                {"results": [{"name": "X", "country": "Y", "latitude": 0, "longitude": 0}]}
            )
        return FakeResponse(
            {"current_weather": {"temperature": 10, "windspeed": 1, "weathercode": 9999}}
        )

    monkeypatch.setattr("weather_agent.weather_tool.requests.get", fake_get)

    result = get_weather("X")

    assert result["condition"] == "unknown"


def test_get_weather_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(
        "weather_agent.weather_tool.requests.get",
        lambda url, params, timeout: FakeResponse({}, status_code=500),
    )

    with pytest.raises(requests.HTTPError):
        get_weather("Hyderabad")
