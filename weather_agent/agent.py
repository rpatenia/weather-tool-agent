"""The tool-calling agent.

This talks to Gemini and drives the tool-calling loop by hand (rather than
relying on the SDK's "automatic function calling" convenience wrapper), so
the mechanics are explicit:

  1. Send the user's message plus the get_weather tool's schema.
  2. If Gemini responds with a function_call instead of text, it wants us to
     run the tool -- we execute it ourselves and send the result back as a
     function_response.
  3. Repeat until Gemini responds with plain text (it could ask for more
     than one call, e.g. "compare Paris and Tokyo").

The model never sees real weather data until we hand it back in step 2 --
it only ever decides *when* to call the tool and *what arguments* to use.
"""

from google import genai
from google.genai import types

from weather_agent.weather_tool import WeatherLookupError, get_weather

DEFAULT_MODEL = "gemini-2.5-flash"

GET_WEATHER_DECLARATION = types.FunctionDeclaration(
    name="get_weather",
    description=(
        "Get current weather conditions (temperature, wind speed, sky "
        "conditions) for a city. Call this for any question about current "
        "weather, temperature, or conditions in a specific place -- never "
        "guess or use prior knowledge, since weather changes constantly."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "city": types.Schema(
                type="STRING",
                description=(
                    "The city name, optionally with country for "
                    "disambiguation, e.g. 'Hyderabad' or 'Paris, France'."
                ),
            ),
        },
        required=["city"],
    ),
)

WEATHER_TOOL = types.Tool(function_declarations=[GET_WEATHER_DECLARATION])

SYSTEM_INSTRUCTION = (
    "You are a weather assistant. You have no built-in knowledge of current "
    "weather anywhere in the world -- you MUST call the get_weather function "
    "to get real, current data before answering any question about weather, "
    "temperature, or conditions. Never invent or guess weather data. If the "
    "user asks something unrelated to weather, answer normally without "
    "calling the tool."
)


class WeatherAgent:
    """A Gemini chat session pre-configured with the get_weather tool."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        client = genai.Client(api_key=api_key)
        self._chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                tools=[WEATHER_TOOL],
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

    def ask(self, user_message: str) -> str:
        """Send one user message through the tool-calling loop, return the reply."""
        response = self._chat.send_message(user_message)

        # Gemini may request zero, one, or several tool calls before it's
        # ready to answer in plain text -- loop until it stops asking.
        while response.function_calls:
            reply_parts = [
                types.Part.from_function_response(
                    name=call.name,
                    response=self._dispatch(call.name, dict(call.args)),
                )
                for call in response.function_calls
            ]
            response = self._chat.send_message(reply_parts)

        return response.text or "(no response)"

    @staticmethod
    def _dispatch(name: str, args: dict) -> dict:
        """Run the tool Gemini asked for and normalize the result to a dict.

        Errors are returned as a {"error": ...} dict rather than raised, so
        Gemini sees them as a function_response and can explain the failure
        to the user in natural language instead of the CLI crashing.
        """
        if name != "get_weather":
            return {"error": f"Unknown tool '{name}'."}
        try:
            return get_weather(**args)
        except WeatherLookupError as exc:
            return {"error": str(exc)}
