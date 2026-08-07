"""Tests for the agent's tool-dispatch logic.

These test WeatherAgent._dispatch directly as a plain function -- it's a
staticmethod with no LLM client involved, so no Gemini API key or network
access is needed. The end-to-end "does the model actually call the tool"
behavior is verified manually (see README) since that requires a live LLM.
"""

from weather_agent import agent as agent_module
from weather_agent.weather_tool import WeatherLookupError


def test_dispatch_calls_get_weather(monkeypatch):
    monkeypatch.setattr(
        agent_module, "get_weather", lambda city: {"city": city, "temperature_c": 20}
    )

    result = agent_module.WeatherAgent._dispatch("get_weather", {"city": "Paris"})

    assert result == {"city": "Paris", "temperature_c": 20}


def test_dispatch_wraps_weather_lookup_error(monkeypatch):
    def raise_error(city):
        raise WeatherLookupError(f"no such place: {city}")

    monkeypatch.setattr(agent_module, "get_weather", raise_error)

    result = agent_module.WeatherAgent._dispatch("get_weather", {"city": "Nowhere"})

    assert result == {"error": "no such place: Nowhere"}


def test_dispatch_rejects_unknown_tool_name():
    result = agent_module.WeatherAgent._dispatch("delete_all_files", {})

    assert "error" in result
