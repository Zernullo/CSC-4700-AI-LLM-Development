import os
import json
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
import requests


# Config / Environment
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # override via env if desired
MAX_STEPS = 5  # guardrail to avoid infinite loops

load_dotenv('../../.env')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Tool Schema (OpenAI style) -------------------------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Retrieve current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name with optional country code, e.g., 'Paris, FR'."
                    },
                    "units": {
                        "type": "string",
                        "description": "Units for temperature.",
                        "enum": ["metric", "imperial"],
                        "default": "metric"
                    }
                },
                "required": ["city"],
                "additionalProperties": False
            }
        }
    }
]


# Weather API Backend (Executor) --------------------------------------------------------------------------------------
def get_weather(city: str, units: str = "metric") -> Dict[str, Any]:
    """
    Executor for the 'get_weather' tool. Calls Open-Meteo API for real weather data.

    :param city: name of the city
    :param units: units for the resulting data (metric or imperial)
    :return: dictionary output of tool call
    """
    
    if not city or not isinstance(city, str):
        return {"error": "INVALID_ARGUMENT", "message": "city must be a non-empty string"}

    if units not in ("metric", "imperial"):
        return {"error": "INVALID_ARGUMENT", "message": "units must be 'metric' or 'imperial'"}

    try:
        # Geocode the city name using Open-Meteo's geocoding API
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
        geo_response = requests.get(geo_url, params=geo_params)
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return {"error": "CITY_NOT_FOUND", "message": f"City '{city}' not found."}

        location = geo_data["results"][0]
        latitude = location["latitude"]
        longitude = location["longitude"]
        city_name = location.get("name", city)

        # Get weather data from Open-Meteo
        weather_url = "https://api.open-meteo.com/v1/forecast"
        temp_unit = "fahrenheit" if units == "imperial" else "celsius"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": temp_unit
        }
        weather_response = requests.get(weather_url, params=weather_params)
        weather_data = weather_response.json()

        current = weather_data.get("current", {})
        temp = current.get("temperature_2m")
        out_units = "F" if units == "imperial" else "C"

        return {
            "city": city_name,
            "temp": temp,
            "units": out_units,
            "conditions": "See weather_code for details",
            "weather_code": current.get("weather_code"),
            "source": "open-meteo-api"
        }
    except Exception as e:
        return {"error": "API_ERROR", "message": str(e)}


# Map tool name -> executor
EXECUTORS = {
    "get_weather": get_weather,
}


# Controller Utilities -------------------------------------------------------------------------------------------------
def run_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch to the correct executor with minimal validation.
    :param name: name of the function to be called
    :param arguments: arguments to the function
    :return: dictionary function output or error message
    """
    fn = EXECUTORS.get(name)
    if fn is None:
        return {"error": "TOOL_NOT_FOUND", "message": f"No tool registered as '{name}'."}

    try:
        # Extract typed args safely
        city = arguments.get("city")
        units = arguments.get("units", "metric")
        return fn(city=city, units=units)
    except Exception as e:
        return {"error": "TOOL_RUNTIME_ERROR", "message": str(e)}


def chat_once(messages, tools=None, tool_choice: Optional[str] = None):
    """
    Make a single Chat Completions call. Returns the raw response object.
    """
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,  # None lets model decide, "auto" also works
        temperature=0.2,
        max_tokens=300
    )


# Agent Loop -----------------------------------------------------------------------------------------------------------
def weather_agent(user_query: str) -> str:
    """
    This is a simple controller loop:
      1) Send system+user + tool catalog to model
      2) If model requests a tool, run it and return tool result
      3) Feed tool result back; repeat until final answer
    :param user_query: the user's string query
    :return: model's response
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful weather assistant. "
                "Use the provided tools when needed. "
                "Answer clearly and concisely."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    for _ in range(MAX_STEPS):
        resp = chat_once(messages, tools=TOOLS, tool_choice="auto")
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            # 1) Append the assistant message WITH tool_calls first
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    } for tc in tool_calls
                ],
                "content": None,
            })

            # 2) Execute each tool call and append a matching 'tool' message
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,      # must match the assistant tool_call id
                    "name": name,
                    "content": json.dumps(result)  # must be a string
                })

            # 3) Loop again so the model can read the tool results
            continue

        # No tool call, final assistant answer
        return (msg.content or "").strip()

    return "I couldn't complete the request within the step budget. Please try again."


# Command line interface -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser("argument parser for weather bot")
    parser.add_argument("m", type=str, help="query to model")
    args = parser.parse_args()

    if len(args.m) < 2:
        print("Usage: python weather_agent.py \"What's the weather in Paris?\"")
        exit(1)

    query = args.m

    answer = weather_agent(query)
    print("\nAssistant:\n", answer)
