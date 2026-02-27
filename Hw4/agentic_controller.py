"""
CSC 4700 Homework 4: Build mini ClawdBot

Author: Daniel Guo (Modify Code - 'Student Complete' Sections)
Instructor: Dr. Keith Mills
TA: Samuel Hildebrand (Skeleton Code)

In this assignment we are building a mini agentic AI system.
This agent doesn't just answer questions but also decides which tools are used based on what you ask it.
The agent has a "controller loop" that maintains state, calls a planner LLM to decide the next action, executes that action (with validation and error handling), and then updates its state with the new evidence before repeating.

References:
Google and Claude were used to clarify syntax, standard library usage, and API interactions.
openai_function_calling_works.py from the example code was used as a reference for how to structure the tool catalog, validation, and execution.
Dr. Mills and Samuel Hildebrand was built the structure of the agent.

Run:
  python agentic_controller.py
"""

# Imports & Setup ------------------------------------------------------------------------------------------------------

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from jsonschema import Draft202012Validator
from dotenv import load_dotenv
from openai import OpenAI
import hashlib
import json
import os
import time
# Suppress unused-import warning: random is available for future stochastic tool selection
import random
import chromadb
import requests

# Load environment variables
load_dotenv('.env')  # .env is in the same directory as the script
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool Catalog ---------------------------------------------------------------------------------------------------------
# Each entry maps a tool name to its JSON Schema for argument validation and planner guidance.
# The tool catalog provides precise JSON Schemas for arguments. This helps to prevent the model from inventing fields
# or tools, and helps validation & auto-repair.

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # Tool 1: a weather tool that retrieves current conditions for a given city
    "weather.get_current": {
        "type": "object",
        "description": "Get the current weather",
        "properties": {
            "city":  {"type": "string", "minLength": 1},
            "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"}
        },
        "required": ["city"],
        "additionalProperties": False
    },
    # Tool 2: knowledge-base search tool for querying the Chroma vector database
    "kb.search": {
        "type": "object",
        "description": "search a knowledge base for information",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "k":     {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}
        },
        "required": ["query"],
        "additionalProperties": False
    },
    # Tool 3: read_mem reads the contents of Memory.md into the agent's working context
    "read_mem": {
        "type": "object",
        "description": "Read the contents of the agent's working memory. Returns the current contents of "
                       "Memory.md, which serves as persistent cross-session memory. Use this to recall "
                       "previously stored facts before answering.",
        "properties": {},  # No arguments needed to read memory
        "required": [],
        "additionalProperties": False
    },
    # Tool 4: write_mem appends new information to Memory.md for future recall
    "write_mem": {
        "type": "object",
        "description": "Append a piece of information to Memory.md for future recall across sessions. "
                       "Use this to store facts the user wants the agent to remember.",
        "properties": {
            # content is the string to append; must be non-empty
            "content": {"type": "string", "minLength": 1}
        },
        "required": ["content"],
        "additionalProperties": False
    },
    # Tool 5: iss.location fetches the real-time position of the International Space Station
    "iss.location": {
        "type": "object",
        "description": "Get the current latitude and longitude of the International Space Station (ISS) "
                       "in real time. No arguments required.",
        "properties": {},  # No arguments needed; position is always live
        "required": [],
        "additionalProperties": False
    }
}

# Optional hints: rough latency/cost so planner can reason about budgets.
# Values are estimates based on typical API response times and token usage.
TOOL_HINTS: Dict[str, Dict[str, Any]] = {
    "weather.get_current": {"avg_ms": 400, "avg_tokens": 50},
    "kb.search":           {"avg_ms": 120, "avg_tokens": 30},
    "read_mem":            {"avg_ms": 5,   "avg_tokens": 200},
    "write_mem":           {"avg_ms": 5,   "avg_tokens": 50},
    "iss.location":        {"avg_ms": 100, "avg_tokens": 30},
}

# Controller State -----------------------------------------------------------------------------------------------------


@dataclass
class StepRecord:
    """Telemetry for each executed step (action)."""

    action: str                  # tool name or 'answer'
    args: Dict[str, Any]         # arguments supplied to the tool
    ok: bool                     # success flag from executor
    latency_ms: int              # wall-clock latency in milliseconds
    info: Dict[str, Any] = field(default_factory=dict)  # normalized payload from tool


@dataclass
class ControllerState:
    """Mutable task state carried through the controller loop."""

    goal: str                    # user task/goal for this session
    history_summary: str = ""    # compact rolling summary (LLM-generated after each step)
    tool_trace: List[StepRecord] = field(default_factory=list)
    tokens_used: int = 0         # simple token accounting across all LLM calls
    cost_cents: float = 0.0      # simple cost accounting in cents
    steps_taken: int = 0         # how many actions have been executed so far
    last_observation: str = ""   # short feedback string from the most recent step
    done: bool = False           # termination flag; set True when final answer is produced


# Budgets & Accounting -------------------------------------------------------------------------------------------------
# Hard ceilings to avoid runaway cost or infinite loops
MAX_STEPS = 8
MAX_TOKENS = 20_000
MAX_COST_CENTS = 75.0


def within_budget(s: ControllerState) -> bool:
    """
    Check hard ceilings for steps, tokens, and cost.

    :param s: instance of ControllerState
    :return: True if still within budget, False if any ceiling is exceeded
    """
    return (
        s.steps_taken < MAX_STEPS and
        s.tokens_used < MAX_TOKENS and
        s.cost_cents < MAX_COST_CENTS
    )


def record_usage(s: ControllerState, usage) -> None:
    """
    Update token/cost counters using the response.usage object if available.
    This is a simplified accounting model for demonstration purposes.

    :param s: instance of ControllerState object
    :param usage: a response.usage object from OpenAI model response
    :return: None
    """
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    total = pt + ct
    s.tokens_used += total
    # gpt-4o-mini pricing: $0.25 per 1M tokens (as of 2025)
    s.cost_cents += total * 0.25 / 1E4


# Loop Detection -------------------------------------------------------------------------------------------------------
# Detect repeated (action, args) pairs to avoid "stuck" ReAct oscillations.
LAST_ACTIONS = deque(maxlen=3)


def fingerprint_action(action: str, args: Dict[str, Any]) -> str:
    """
    Hash the tool call pair (action, args) to compare recent moves.

    :param action: the action the model selected
    :param args: the arguments the model selected for the action
    :return: A sha256 hex digest string
    """
    # Serialize action and args deterministically, then hash to produce a fixed-length fingerprint
    blob = json.dumps({"a": action, "x": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def looks_stuck(action: str, args: Dict[str, Any]) -> bool:
    """
    Return True if the last N actions are identical (loop detected).

    :param action: the action the model selected
    :param args: the arguments the model selected for the action
    :return: True if this action matches all recent actions, False otherwise
    """
    fp = fingerprint_action(action, args)
    LAST_ACTIONS.append(fp)
    return (
        len(LAST_ACTIONS) == LAST_ACTIONS.maxlen and
        len(set(LAST_ACTIONS)) == 1
    )


# Arg Validation & Repair ----------------------------------------------------------------------------------------------

def validate_args(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate args against the JSON Schema for the given tool.

    :param tool_name: the name of the tool that the model selected
    :param args: the arguments the model selected for that tool
    :return: (True, None) if valid, (False, error_message) if not
    """
    schema = TOOL_SCHEMAS[tool_name]
    validator = Draft202012Validator(schema)
    # Sort errors by path for deterministic first-error reporting
    errors = sorted(validator.iter_errors(args), key=lambda e: e.path)
    if errors:
        e = errors[0]
        path = "/".join([str(p) for p in e.path]) or "(root)"
        return False, f"Invalid arguments at {path}: {e.message}"
    return True, None


def repair_args_with_llm(tool_name: str, bad_args: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """
    Ask the LLM to fix only the invalid parts to satisfy the JSON Schema.
    Enforces JSON-only output and re-validates after repair.

    :param tool_name: name of the selected tool
    :param bad_args: dictionary of bad arguments provided by the model
    :param error_msg: the error message produced by the validator
    :return: corrected arguments dictionary
    """
    schema = TOOL_SCHEMAS[tool_name]
    dev = (
        "You fix JSON arguments to match a JSON Schema. "
        "Return VALID JSON only—no prose, no code fences, no comments."
    )
    user = json.dumps({
        "tool_name": tool_name,
        "schema": schema,
        "invalid_args": bad_args,
        "validator_error": error_msg
    })
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},  # force JSON-only output
        messages=[
            {"role": "developer", "content": dev},
            {"role": "user", "content": user}
        ]
    )
    return json.loads(resp.choices[0].message.content)


# History Summarization ------------------------------------------------------------------------------------------------

def update_summary(state: ControllerState, new_evidence: str) -> None:
    """
    Compress the prior summary and new evidence into a short rolling memory.
    Keeps context window small while preserving key facts and decisions.

    :param state: instance of ControllerState
    :param new_evidence: the observation string returned by the most recent tool call
    :return: None
    """
    sys = (
        "Compress facts and decisions into <=120 tokens. Keep IDs and key numbers. "
        "Do not include anything that is unnecessary, only things that are strictly useful for the goal."
    )
    user = json.dumps({
        "prior_summary": state.history_summary,
        "new_evidence": new_evidence
    })
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": sys},
            {"role": "user", "content": user}
        ]
    )
    content = resp.choices[0].message.content.strip()
    state.history_summary = content
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)


# Planner --------------------------------------------------------------------------------------------------------------

def plan_next_action(state: ControllerState) -> Tuple[str, Dict[str, Any], str]:
    """
    Ask the LLM to pick ONE next action toward the goal:
      - a known tool from TOOL_SCHEMAS with valid arguments, OR
      - the literal string 'answer' when enough evidence exists to respond.

    :param state: instance of ControllerState
    :return: (action, args, rationale) tuple
    """
    # Build the tool spec list to send to the planner, including schema, budget hints, and examples
    tool_specs = []
    for name, schema in TOOL_SCHEMAS.items():
        spec = {
            "name": name,
            "schema": schema,  # full JSON Schema so planner knows valid argument shapes
            "budget_hint": {
                "avg_ms": TOOL_HINTS[name]["avg_ms"],
                "avg_tokens": TOOL_HINTS[name]["avg_tokens"],
            },
            # Few-shot examples help the planner produce schema-compliant arguments
            "examples": {
                "weather.get_current": [
                    {"city": "Paris", "units": "metric"},
                    {"city": "New York"}  # units defaults to metric via schema
                ],
                "kb.search": [
                    {"query": "VPN policy for contractors", "k": 3}
                ],
                "read_mem": [
                    {}  # no arguments needed
                ],
                "write_mem": [
                    {"content": "User's name is Alice."}
                ],
                "iss.location": [
                    {}  # no arguments needed
                ]
            }.get(name, [])
        }
        tool_specs.append(spec)

    dev = (
        "You are a planner. Choose ONE next action toward the goal. Do not call actions towards "
        "information already contained in the history summary provided below.\n"
        "Use ONLY tools from `tool_catalog` OR choose 'answer' if you can respond now.\n"
        "You can only answer with information provided by the tools. "
        "When using a tool, produce arguments that VALIDATE against its JSON Schema.\n"
        "Allowed output format (JSON only):\n"
        '{"action":"<tool_name|answer>","args":{...}, "rationale":"<brief reason>"} '
        "You have memory that you can access with the read_mem and write_mem tools."
    )

    user = json.dumps({
        "goal": state.goal,
        "budget": {
            "steps_remaining": MAX_STEPS - state.steps_taken,
            "tokens_remaining": MAX_TOKENS - state.tokens_used,
            "cost_cents_remaining": round(MAX_COST_CENTS - state.cost_cents, 2)
        },
        "history_summary": state.history_summary,
        "tool_catalog": tool_specs,
        "last_observation": state.last_observation
    })

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},  # enforce JSON output from planner
        messages=[
            {"role": "developer", "content": dev},
            {"role": "user", "content": user}
        ]
    )
    obj = json.loads(resp.choices[0].message.content)
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)
    return obj["action"], obj.get("args", {}), obj.get("rationale", "")


# Executor -------------------------------------------------------------------------------------------------------------

def execute_action(action: str, args: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any], int]:
    """
    Execute the selected action with validation, repair, retries, and error handling.

    :param action: tool name selected by the planner
    :param args: arguments selected by the planner for the tool
    :return: (ok, observation_text, normalized_payload, latency_ms)
    """
    t0 = time.time()

    # 'answer' is a virtual tool signaling that the agent should synthesize the final answer
    if action == "answer":
        obs = "Ready to synthesize final answer from working memory and evidence."
        return True, obs, {}, int((time.time() - t0) * 1000)

    # Guard: reject calls to tools not registered in the catalog
    if action not in TOOL_SCHEMAS:
        return False, f"Unknown tool: {action}", {}, int((time.time() - t0) * 1000)

    # Step 1: validate arguments against the tool's JSON Schema
    ok, msg = validate_args(action, args)
    if not ok:
        # Step 2: one-shot LLM repair attempt; re-validate the result
        fixed = repair_args_with_llm(action, args, msg)
        ok2, msg2 = validate_args(action, fixed)
        if not ok2:
            return False, f"Arg repair failed: {msg2}", {}, int((time.time() - t0) * 1000)
        args = fixed

    # Step 3: dispatch to the correct tool executor
    try:
        if action == "weather.get_current":
            city = args["city"]
            units = args.get("units", "metric")

            # Step 1: geocode city name to lat/lon using Open-Meteo's free geocoding API
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
            geo_response = requests.get(geo_url, params=geo_params)
            geo_data = geo_response.json()

            if not geo_data.get("results"):
                return False, f"City '{city}' not found.", {}, int((time.time() - t0) * 1000)

            location = geo_data["results"][0]
            latitude = location["latitude"]
            longitude = location["longitude"]
            city_name = location.get("name", city)

            # Step 2: fetch current temperature and weather code from Open-Meteo forecast API
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

            payload = {
                "city": city_name,
                "temp": temp,
                "units": out_units,
                "conditions": "See weather_code for details",
                "weather_code": current.get("weather_code"),
                "source": "open-meteo-api"
            }
            obs = f"Weather in {payload['city']}: {payload['temp']}° {out_units} ({payload['conditions']})"
            return True, obs, payload, int((time.time() - t0) * 1000)

        elif action == "kb.search":
            try:
                # Connect to the persistent Chroma vector database stored alongside this script
                db_path = os.path.join(os.path.dirname(__file__), "kb_chroma_db")
                chroma_client = chromadb.PersistentClient(path=db_path)
                collection = chroma_client.get_or_create_collection("handbook")

                query = args["query"]
                k = int(args.get("k", 5))
                results = collection.query(query_texts=[query], n_results=k)

                # Zip documents and metadata together into snippet dicts
                snippets = []
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    snippets.append({"snippet": doc, "meta": meta})

                obs = f"Retrieved {len(snippets)} snippets: " + " | ".join(
                    s["snippet"][:100] for s in snippets
                )
                return True, obs, {"results": snippets}, int((time.time() - t0) * 1000)
            except Exception as e:
                return False, f"KB search error: {e}", {}, int((time.time() - t0) * 1000)

        elif action == "read_mem":
            # Resolve Memory.md path relative to this script, not the working directory
            memory_path = os.path.join(os.path.dirname(__file__), "Memory.md")
            try:
                # Create the file if it does not yet exist
                if not os.path.exists(memory_path):
                    with open(memory_path, "w", encoding="utf-8") as f:
                        f.write("")

                with open(memory_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Return content as observation so it enters the planner's working context
                obs = f"Memory contents:\n{content}"
                payload = {"content": content}
                return True, obs, payload, int((time.time() - t0) * 1000)

            except Exception as e:
                return False, f"Memory read error: {e}", {}, int((time.time() - t0) * 1000)

        elif action == "write_mem":
            # Resolve Memory.md path relative to this script, not the working directory
            memory_path = os.path.join(os.path.dirname(__file__), "Memory.md")
            try:
                content = args["content"]

                # Append rather than overwrite so memory accumulates across sessions
                with open(memory_path, "a", encoding="utf-8") as f:
                    f.write(content + "\n")

                obs = f"Appended {len(content)} characters to memory."
                payload = {"written": len(content)}
                return True, obs, payload, int((time.time() - t0) * 1000)

            except Exception as e:
                return False, f"Memory write error: {e}", {}, int((time.time() - t0) * 1000)

        elif action == "iss.location":
            try:
                # Open Notify provides a free, no-auth endpoint for real-time ISS position
                response = requests.get("http://api.open-notify.org/iss-now.json", timeout=5)
                data = response.json()
                position = data["iss_position"]
                latitude = position["latitude"]
                longitude = position["longitude"]

                payload = {"latitude": latitude, "longitude": longitude}
                obs = f"Current ISS location: Latitude {latitude}, Longitude {longitude}"
                return True, obs, payload, int((time.time() - t0) * 1000)

            except Exception as e:
                return False, f"ISS location error: {e}", {}, int((time.time() - t0) * 1000)

        else:
            # Safety fallback: no executor is wired for this tool name
            return False, f"No executor bound for tool: {action}", {}, int((time.time() - t0) * 1000)

    except Exception as e:
        # Catch-all for unexpected runtime errors in any tool executor
        return False, f"Tool error: {type(e).__name__}: {e}", {}, int((time.time() - t0) * 1000)


# Final Synthesis ------------------------------------------------------------------------------------------------------

def synthesize_answer(state: ControllerState) -> str:
    """
    Compose the final answer using the compact working summary accumulated in state.history_summary.
    The full raw trace can be logged separately for debugging.

    :param state: instance of ControllerState
    :return: final answer string from the model
    """
    sys = (
        "Your goal is to produce a final answer to a goal (likely a question) using only "
        "evidence provided in the working summary."
    )
    # Constrain output length to avoid runaway generation in the final synthesis step
    user = (
        f"Goal: {state.goal}\n\n"
        f"Working summary:\n{state.history_summary}\n\n"
        f"Produce the final answer in <= 200 tokens."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": sys},
            {"role": "user", "content": user}
        ]
    )
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)
    return resp.choices[0].message.content.strip()


# Controller Loop ------------------------------------------------------------------------------------------------------

def run_agent(goal: str) -> str:
    """
    Main controller loop:
      while budgets remain and not done:
        1) Plan next action (tool or 'answer') using the planner LLM
        2) Loop detection guard to prevent oscillation
        3) Execute the chosen action with validation and repair
        4) Update rolling summary with new evidence
        5) If 'answer', synthesize final output and stop

    :param goal: the user's query string
    :return: final answer string or budget-exhausted message
    """
    state = ControllerState(goal=goal)

    while within_budget(state) and not state.done:
        # Ask the planner to choose the next action
        action, args, rationale = plan_next_action(state)
        print(f"Action selected: {action}\n\targuments: {args}\n\trationale: {rationale}")

        # Prevent infinite ReAct loops by comparing fingerprints of recent actions
        if looks_stuck(action, args):
            print("\tdetected being stuck in loop...")
            state.last_observation = "Loop detected: revise plan with a different next action."
            # Do not increment steps or execute; let planner try again with new context
            continue

        # Execute the chosen action (or 'answer' pseudo-tool)
        ok, obs, payload, ms = execute_action(action, args)
        print(f"\t\ttool payload: {payload}")

        # Record step telemetry for debugging and cost tracking
        state.steps_taken += 1
        state.tool_trace.append(StepRecord(
            action=action,
            args=args,
            ok=ok,
            latency_ms=ms,
            info=payload
        ))

        # Feed the short observation back to the planner for the next iteration
        state.last_observation = obs

        # Compress new evidence into the rolling working memory summary
        update_summary(state, f"{action}({args}) -> {obs}")

        # If planner signaled 'answer', produce the final answer and exit the loop
        if action == "answer" and ok:
            final = synthesize_answer(state)
            state.done = True
            return final

        # If a tool failed, the planner sees the observation and can pivot on the next step
    print(within_budget(state), state.done)
    # Reached here only if budgets are exhausted or the agent made no progress
    return "Stopped: budget exhausted or no progress."


# Entry Point ----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse a single positional argument: the user's query string
    # Example usage:
    #   python agentic_controller.py "What is today's weather in Baton Rouge?"
    #   python agentic_controller.py "Where is the ISS right now?"
    #   python agentic_controller.py "My name is Daniel, please remember it."
    parser = argparse.ArgumentParser(description="Agentic Controller")
    parser.add_argument("query", type=str, help="Your query to the agent")
    parsed = parser.parse_args()

    print("\n--- Running Agent ---\n")
    answer = run_agent(parsed.query)
    print("\n--- Final Answer ---\n")
    print(answer)