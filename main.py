"""CLI entrypoint for the weather agent.

Usage:
    python main.py                                    # interactive REPL
    python main.py "What's the weather in Hyderabad?"  # one-shot question
"""

import os
import sys

from dotenv import load_dotenv

# Windows consoles often default stdout to a legacy codepage (cp1252) that
# can't encode "°", turning Gemini's replies into mojibake (e.g. "27.7°C"
# -> "27.7?C"). Force UTF-8 on stdout/stderr; harmless on platforms that are
# already UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from weather_agent.agent import DEFAULT_MODEL, WeatherAgent


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "GEMINI_API_KEY is not set.\n"
            "Copy .env.example to .env and add a key from "
            "https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        sys.exit(1)

    agent = WeatherAgent(api_key=api_key, model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))

    # One-shot mode: `python main.py "..."`.
    if len(sys.argv) > 1:
        print(agent.ask(" ".join(sys.argv[1:])))
        return

    # Interactive REPL.
    print("Weather assistant -- ask about the weather anywhere. Type 'exit' to quit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        print(agent.ask(question))


if __name__ == "__main__":
    main()
