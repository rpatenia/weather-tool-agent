# Weather Agent

A small CLI that answers natural-language weather questions (e.g. *"What's
the weather in Hyderabad?"*) by having an LLM call a real weather-fetching
tool, rather than guessing from its training data.

## Tech stack

- **LLM: Google Gemini** (`gemini-2.5-flash`), via the official
  [`google-genai`](https://pypi.org/project/google-genai/) SDK. Free tier
  via [Google AI Studio](https://aistudio.google.com/apikey) is more than
  enough for this.
- **Weather data: [Open-Meteo](https://open-meteo.com/)**. Free, no API key
  needed. A geocoding endpoint turns a free-text city name into
  coordinates; a forecast endpoint returns current conditions for those
  coordinates.
- **Python 3.12**, `requests` for HTTP, `python-dotenv` for loading the API
  key from a `.env` file, `pytest` for tests. No web framework — this is a
  CLI, per the task's own "no UI needed."

## How it works

```
weather_agent/
  weather_tool.py   # get_weather(city) -> dict — plain function, no LLM code in it
  agent.py          # WeatherAgent — owns the Gemini chat + the tool-calling loop
main.py             # CLI: one-shot ("python main.py '...'") or interactive REPL
tests/
  test_weather_tool.py  # weather_tool tested with faked HTTP responses
  test_agent.py         # dispatch logic tested without touching the LLM
```

The two concerns are deliberately kept apart: `weather_tool.py` doesn't know
an LLM exists, and `agent.py` doesn't know how weather is actually fetched
— it just calls `get_weather(city)` and gets a dict back. That's what makes
both halves testable without a live API key.

### The tool-calling loop

`agent.py` declares one tool for Gemini, `get_weather(city: str)`, with a
JSON-schema description of its single argument. Each turn:

1. The user's message (plus the tool schema and a system instruction telling
   the model it has *no* built-in weather knowledge and must call the tool)
   is sent to Gemini.
2. If the question needs weather data, Gemini doesn't reply with text — it
   replies with a `function_call` part naming `get_weather` and the `city`
   argument it inferred from the question (e.g. extracting `"Hyderabad"`
   from *"What's the weather in Hyderabad?"*).
3. The code (not the model) executes `get_weather(city)` for real, against
   Open-Meteo, and sends the result back to Gemini as a `function_response`.
4. Gemini reads the real data and produces the final natural-language reply.

This loop repeats until Gemini stops asking for function calls — so a
question like *"compare the weather in Paris and Tokyo"* triggers two calls
in the same turn and both results come back before the final answer.

The loop is written by hand (not via the SDK's automatic-function-calling
helper) specifically so this request → tool-call → tool-result → response
cycle is visible in the code rather than hidden inside a library call.

### Error handling

If a city can't be geocoded, `get_weather` raises `WeatherLookupError`.
`agent.py` catches that and hands Gemini `{"error": "..."}` as the function
result instead of letting the exception crash the CLI — Gemini then explains
the problem to the user in plain language (e.g. "I couldn't find a city
called ...") instead of the program dying with a traceback.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements-dev.txt   # includes pytest; use requirements.txt for runtime-only

cp .env.example .env
# edit .env, set GEMINI_API_KEY to a key from https://aistudio.google.com/apikey
```

## Run it

```bash
# one-shot
python main.py "What's the weather in Hyderabad?"

# interactive
python main.py
> What's the weather like in Tokyo right now?
> exit
```

## Test it

```bash
pytest
```

All tests fake their HTTP/LLM calls, so `pytest` runs without a `GEMINI_API_KEY`
or network access. The one thing not covered by an automated test is
"does Gemini actually decide to call the tool" — that needs a live LLM call,
so it's verified manually by running the CLI (see above) rather than
mocking Gemini's decision-making, which would just test the mock.

## Design decisions worth calling out

- **Open-Meteo over OpenWeatherMap**: no signup/API key needed, so the repo
  is runnable immediately with just a Gemini key — one fewer account to
  create to try it. Trade-off: OpenWeatherMap's key-activation-delay problem
  is avoided, but Open-Meteo's geocoding is a plain nearest-name match, no
  fuzzy disambiguation UI for ambiguous city names (there are several
  "Hyderabad"s; the tool takes the top geocoding match, and an optional
  `"City, Country"` string disambiguates when needed).
- **Manual tool-calling loop over SDK auto-function-calling**: Gemini's SDK
  can register a plain Python function and dispatch it automatically. That's
  less code, but it hides exactly the mechanic this task is about. Doing it
  by hand costs ~15 lines and makes the request/response cycle inspectable
  and explainable.
- **Errors returned to the model, not raised to the user**: a lookup failure
  becomes a `function_response` with an `error` field, so the *model*
  explains the failure conversationally instead of the CLI printing a raw
  traceback.
