"""
CSC 4700 Homework 4: Build mini ClawdBot

Author: Daniel Guo (Modify Code - Student Complete Sections)
Instructor: Dr. Keith Mills
TA: Samuel Hildebrand (Skeleton Code)

This assignment builds a mini agentic AI system. The agent decides
which tools to use based on the query, maintaining state across a
controller loop that plans, executes, and summarizes each step.

References:
- Google and Claude: syntax, standard library, API interactions.
- openai_function_calling_works.py: tool catalog and validation structure.
- Dr. Mills and Samuel Hildebrand: agent skeleton.

Run:
  python agentic_controller.py "your query here"
"""

# Imports & Setup ----------------------------------------------------------

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
import random  # available for future stochastic tool selection
import chromadb
import requests

# Load environment variables from the same directory as the script
load_dotenv('.env')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool Catalog -------------------------------------------------------------
# Each tool name maps to a JSON Schema for argument validation and planner
# guidance. Strict schemas prevent the model from inventing fields or tools.

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # Tool 1: retrieves current weather conditions for a given city
    "weather.get_current": {
        "type": "object",
        "description": "Get the current weather for a city.",
        "properties": {
            "city": {"type": "string", "minLength": 1},
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "default": "metric"
            }
        },
        "required": ["city"],
        "additionalProperties": False
    },
    # Tool 2: vector search over the Chroma knowledge base
    "kb.search": {
        "type": "object",
        "description": "Search a knowledge base for information.",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5
            }
        },
        "required": ["query"],
        "additionalProperties": False
    },
    # Tool 3: reads Memory.md contents into the agent's working context
    "read_mem": {
        "type": "object",
        "description": (
            "Read the agent's persistent memory from Memory.md. "
            "Use this to recall facts stored in previous sessions."
        ),
        "properties": {},  # no arguments required
        "required": [],
        "additionalProperties": False
    },
    # Tool 4: appends new information to Memory.md for future sessions
    "write_mem": {
        "type": "object",
        "description": (
            "Append a fact to Memory.md for cross-session recall. "
            "Use this when the user asks the agent to remember something."
        ),
        "properties": {
            # content: the string to persist; must be non-empty
            "content": {"type": "string", "minLength": 1}
        },
        "required": ["content"],
        "additionalProperties": False
    },
    # Tool 5: fetches the real-time position of the ISS
    "iss.location": {
        "type": "object",
        "description": (
            "Get the current latitude and longitude of the "
            "International Space Station in real time."
        ),
        "properties": {},  # no arguments required; position is always live
        "required": [],
        "additionalProperties": False
    }
}

# Rough latency/token estimates so the planner can reason about budgets.
# Values are based on typical API response times and average token usage.
TOOL_HINTS: Dict[str, Dict[str, Any]] = {
    "weather.get_current": {"avg_ms": 400, "avg_tokens": 50},
    "kb.search":           {"avg_ms": 120, "avg_tokens": 30},
    "read_mem":            {"avg_ms": 5,   "avg_tokens": 200},
    "write_mem":           {"avg_ms": 5,   "avg_tokens": 50},
    "iss.location":        {"avg_ms": 100, "avg_tokens": 30},
}

# Controller State ---------------------------------------------------------


@dataclass


class StepRecord:
    """Telemetry recorded for each executed step (action)."""

    action: str                 # tool name or 'answer'
    args: Dict[str, Any]        # arguments supplied to the tool
    ok: bool                    # success flag returned by executor
    latency_ms: int             # wall-clock latency in milliseconds
    info: Dict[str, Any] = field(default_factory=dict)  # tool payload


@dataclass


class ControllerState:
    """Mutable task state carried through the controller loop."""

    goal: str                   # user task/goal for this session
    history_summary: str = ""   # rolling summary, LLM-compressed each step
    tool_trace: List[StepRecord] = field(default_factory=list)
    tokens_used: int = 0        # cumulative token count across all LLM calls
    cost_cents: float = 0.0     # cumulative cost in cents
    steps_taken: int = 0        # number of actions executed so far
    last_observation: str = ""  # feedback string from the most recent step
    done: bool = False          # True once the final answer is produced


# Budgets & Accounting -----------------------------------------------------
# Hard ceilings prevent runaway cost or infinite loops.
MAX_STEPS = 8
MAX_TOKENS = 20_000
MAX_COST_CENTS = 75.0


def within_budget(s: ControllerState) -> bool:
    """
    Check all hard ceilings: steps, tokens, and cost.

    :param s: current ControllerState
    :return: True if within all limits, False if any ceiling is exceeded
    """
    return (
        s.steps_taken < MAX_STEPS
        and s.tokens_used < MAX_TOKENS
        and s.cost_cents < MAX_COST_CENTS
    )


def record_usage(s: ControllerState, usage) -> None:
    """
    Update token and cost counters from a response.usage object.

    :param s: current ControllerState
    :param usage: response.usage object from an OpenAI API response
    :return: None
    """
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    total = pt + ct
    s.tokens_used += total
    # gpt-4o-mini pricing: $0.25 per 1M tokens (as of 2025)
    s.cost_cents += total * 0.25 / 1E4


# Loop Detection -----------------------------------------------------------
# Repeated (action, args) pairs signal a stuck ReAct oscillation.
LAST_ACTIONS = deque(maxlen=3)


def fingerprint_action(action: str, args: Dict[str, Any]) -> str:
    """
    Produce a deterministic hash of the (action, args) pair.

    :param action: tool name selected by the planner
    :param args: arguments selected for that tool
    :return: sha256 hex digest string
    """
    # Sort keys for determinism before hashing
    blob = json.dumps({"a": action, "x": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def looks_stuck(action: str, args: Dict[str, Any]) -> bool:
    """
    Return True if the last N actions are all identical.

    :param action: tool name selected by the planner
    :param args: arguments selected for that tool
    :return: True if a loop is detected, False otherwise
    """
    fp = fingerprint_action(action, args)
    LAST_ACTIONS.append(fp)
    return (
        len(LAST_ACTIONS) == LAST_ACTIONS.maxlen
        and len(set(LAST_ACTIONS)) == 1
    )


# Arg Validation & Repair --------------------------------------------------


def validate_args(
    tool_name: str,
    args: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate args against the registered JSON Schema for the tool.

    :param tool_name: name of the tool to validate against
    :param args: arguments dict produced by the planner
    :return: (True, None) if valid, (False, error_message) otherwise
    """
    schema = TOOL_SCHEMAS[tool_name]
    validator = Draft202012Validator(schema)
    # Sort by path for deterministic first-error selection
    errors = sorted(validator.iter_errors(args), key=lambda e: e.path)
    if errors:
        e = errors[0]
        path = "/".join([str(p) for p in e.path]) or "(root)"
        return False, f"Invalid arguments at {path}: {e.message}"
    return True, None


def repair_args_with_llm(
    tool_name: str,
    bad_args: Dict[str, Any],
    error_msg: str
) -> Dict[str, Any]:
    """
    Ask the LLM to correct invalid arguments to satisfy the JSON Schema.

    :param tool_name: name of the tool whose schema must be satisfied
    :param bad_args: the invalid arguments dict from the planner
    :param error_msg: validation error message to guide the repair
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


# History Summarization ----------------------------------------------------


def update_summary(state: ControllerState, new_evidence: str) -> None:
    """
    Compress the prior summary and new evidence into a rolling memory.
    Keeps the context window small while retaining key facts.

    :param state: current ControllerState
    :param new_evidence: observation string from the most recent tool call
    :return: None
    """
    sys = (
        "Compress facts and decisions into <=120 tokens. "
        "Keep IDs and key numbers. Omit anything not strictly "
        "useful for the goal."
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
    state.history_summary = resp.choices[0].message.content.strip()
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)


# Planner ------------------------------------------------------------------


def plan_next_action(
    state: ControllerState
) -> Tuple[str, Dict[str, Any], str]:
    """
    Ask the LLM to select ONE next action toward the goal.
    Returns a known tool name with arguments, or 'answer' when
    sufficient evidence exists to produce the final response.

    :param state: current ControllerState
    :return: (action, args, rationale) tuple
    """
    # Build tool specs including schema, budget hints, and few-shot examples
    tool_specs = []
    for name, schema in TOOL_SCHEMAS.items():
        spec = {
            "name": name,
            "schema": schema,  # full schema guides valid argument formation
            "budget_hint": {
                "avg_ms": TOOL_HINTS[name]["avg_ms"],
                "avg_tokens": TOOL_HINTS[name]["avg_tokens"],
            },
            # Few-shot examples help the planner form schema-compliant args
            "examples": {
                "weather.get_current": [
                    {"city": "Paris", "units": "metric"},
                    {"city": "New York"}  # units defaults to metric
                ],
                "kb.search": [
                    {"query": "VPN policy for contractors", "k": 3}
                ],
                "read_mem":    [{}],  # no arguments needed
                "write_mem":   [{"content": "User's name is Alice."}],
                "iss.location": [{}],  # no arguments needed
            }.get(name, [])
        }
        tool_specs.append(spec)

    dev = (
        "You are a planner. Choose ONE next action toward the goal. "
        "Do not repeat actions for information already in the summary.\n"
        "Use ONLY tools from `tool_catalog` OR choose 'answer' to respond.\n"
        "Only answer using information provided by the tools.\n"
        "Produce arguments that VALIDATE against the tool JSON Schema.\n"
        "Output format (JSON only):\n"
        '{"action":"<tool_name|answer>",'
        '"args":{...},'
        '"rationale":"<brief reason>"}\n'
        "Use read_mem and write_mem to access persistent memory."
    )

    user = json.dumps({
        "goal": state.goal,
        "budget": {
            "steps_remaining": MAX_STEPS - state.steps_taken,
            "tokens_remaining": MAX_TOKENS - state.tokens_used,
            "cost_cents_remaining": round(
                MAX_COST_CENTS - state.cost_cents, 2
            )
        },
        "history_summary": state.history_summary,
        "tool_catalog": tool_specs,
        "last_observation": state.last_observation
    })

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},  # enforce JSON output
        messages=[
            {"role": "developer", "content": dev},
            {"role": "user", "content": user}
        ]
    )
    obj = json.loads(resp.choices[0].message.content)
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)
    return obj["action"], obj.get("args", {}), obj.get("rationale", "")


# Executor -----------------------------------------------------------------


def execute_action(
    action: str,
    args: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any], int]:
    """
    Dispatch the selected action with validation, repair, and error handling.

    :param action: tool name selected by the planner
    :param args: arguments selected by the planner
    :return: (ok, observation_text, normalized_payload, latency_ms)
    """
    t0 = time.time()

    # 'answer' signals the agent to synthesize a final response
    if action == "answer":
        obs = "Ready to synthesize final answer from evidence."
        return True, obs, {}, int((time.time() - t0) * 1000)

    # Reject calls to tools not registered in the catalog
    if action not in TOOL_SCHEMAS:
        return (
            False,
            f"Unknown tool: {action}",
            {},
            int((time.time() - t0) * 1000)
        )

    # Validate arguments against the tool's JSON Schema
    ok, msg = validate_args(action, args)
    if not ok:
        # Attempt a one-shot LLM repair and re-validate
        fixed = repair_args_with_llm(action, args, msg)
        ok2, msg2 = validate_args(action, fixed)
        if not ok2:
            return (
                False,
                f"Arg repair failed: {msg2}",
                {},
                int((time.time() - t0) * 1000)
            )
        args = fixed

    try:
        if action == "weather.get_current":
            city = args["city"]
            units = args.get("units", "metric")

            # Step 1: geocode city to lat/lon via Open-Meteo geocoding API
            geo_resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": city,
                    "count": 1,
                    "language": "en",
                    "format": "json"
                }
            )
            geo_data = geo_resp.json()

            if not geo_data.get("results"):
                return (
                    False,
                    f"City '{city}' not found.",
                    {},
                    int((time.time() - t0) * 1000)
                )

            loc = geo_data["results"][0]
            city_name = loc.get("name", city)

            # Step 2: fetch temperature and weather code from Open-Meteo
            temp_unit = "fahrenheit" if units == "imperial" else "celsius"
            wx_resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "current": "temperature_2m,weather_code",
                    "temperature_unit": temp_unit
                }
            )
            current = wx_resp.json().get("current", {})
            temp = current.get("temperature_2m")
            out_units = "F" if units == "imperial" else "C"

            payload = {
                "city": city_name,
                "temp": temp,
                "units": out_units,
                "weather_code": current.get("weather_code"),
                "source": "open-meteo-api"
            }
            obs = (
                f"Weather in {city_name}: "
                f"{temp}° {out_units} (code {payload['weather_code']})"
            )
            return True, obs, payload, int((time.time() - t0) * 1000)

        elif action == "kb.search":
            try:
                # Connect to the Chroma DB stored alongside this script
                db_path = os.path.join(
                    os.path.dirname(__file__), "kb_chroma_db"
                )
                chroma_client = chromadb.PersistentClient(path=db_path)
                collection = chroma_client.get_or_create_collection(
                    "handbook"
                )

                query = args["query"]
                k = int(args.get("k", 5))
                results = collection.query(
                    query_texts=[query], n_results=k
                )

                # Combine documents and metadata into snippet dicts
                snippets = [
                    {"snippet": doc, "meta": meta}
                    for doc, meta in zip(
                        results["documents"][0],
                        results["metadatas"][0]
                    )
                ]

                obs = (
                    f"Retrieved {len(snippets)} snippets: "
                    + " | ".join(s["snippet"][:100] for s in snippets)
                )
                return (
                    True, obs,
                    {"results": snippets},
                    int((time.time() - t0) * 1000)
                )
            except Exception as e:
                return (
                    False,
                    f"KB search error: {e}",
                    {},
                    int((time.time() - t0) * 1000)
                )

        elif action == "read_mem":
            # Resolve path relative to this script, not the working dir
            memory_path = os.path.join(
                os.path.dirname(__file__), "Memory.md"
            )
            try:
                # Create the file on first use if it does not exist
                if not os.path.exists(memory_path):
                    with open(memory_path, "w", encoding="utf-8") as f:
                        f.write("")

                with open(memory_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Return content as observation to enter the planner context
                obs = f"Memory contents:\n{content}"
                return (
                    True, obs,
                    {"content": content},
                    int((time.time() - t0) * 1000)
                )
            except Exception as e:
                return (
                    False,
                    f"Memory read error: {e}",
                    {},
                    int((time.time() - t0) * 1000)
                )

        elif action == "write_mem":
            # Resolve path relative to this script, not the working dir
            memory_path = os.path.join(
                os.path.dirname(__file__), "Memory.md"
            )
            try:
                content = args["content"]
                # Append so memory accumulates across sessions
                with open(memory_path, "a", encoding="utf-8") as f:
                    f.write(content + "\n")

                obs = f"Appended {len(content)} characters to memory."
                return (
                    True, obs,
                    {"written": len(content)},
                    int((time.time() - t0) * 1000)
                )
            except Exception as e:
                return (
                    False,
                    f"Memory write error: {e}",
                    {},
                    int((time.time() - t0) * 1000)
                )

        elif action == "iss.location":
            try:
                # Open Notify: free, no-auth endpoint for live ISS position
                resp = requests.get(
                    "http://api.open-notify.org/iss-now.json",
                    timeout=5
                )
                pos = resp.json()["iss_position"]
                lat, lon = pos["latitude"], pos["longitude"]
                payload = {"latitude": lat, "longitude": lon}
                obs = (
                    f"Current ISS location: "
                    f"Latitude {lat}, Longitude {lon}"
                )
                return True, obs, payload, int((time.time() - t0) * 1000)
            except Exception as e:
                return (
                    False,
                    f"ISS location error: {e}",
                    {},
                    int((time.time() - t0) * 1000)
                )

        else:
            # Safety fallback: no executor registered for this tool
            return (
                False,
                f"No executor bound for tool: {action}",
                {},
                int((time.time() - t0) * 1000)
            )

    except Exception as e:
        # Catch-all for unexpected runtime errors in any executor
        return (
            False,
            f"Tool error: {type(e).__name__}: {e}",
            {},
            int((time.time() - t0) * 1000)
        )


# Final Synthesis ----------------------------------------------------------


def synthesize_answer(state: ControllerState) -> str:
    """
    Produce the final answer using only the accumulated working summary.

    :param state: current ControllerState
    :return: final answer string from the model
    """
    sys = (
        "Produce a final answer using only the evidence in the "
        "working summary. Do not invent information."
    )
    # Limit output to avoid runaway generation in the synthesis step
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


# Controller Loop ----------------------------------------------------------


def run_agent(goal: str) -> str:
    """
    Main agentic controller loop.

    Each iteration:
      1. Plans the next action via the planner LLM.
      2. Guards against stuck loops using action fingerprints.
      3. Executes the action with validation and repair.
      4. Updates the rolling summary with new evidence.
      5. Synthesizes and returns the final answer on 'answer'.

    :param goal: the user's query string
    :return: final answer string, or a budget-exhausted message
    """
    state = ControllerState(goal=goal)

    while within_budget(state) and not state.done:
        # Choose the next action
        action, args, rationale = plan_next_action(state)
        print(
            f"Action selected: {action}\n"
            f"\targuments: {args}\n"
            f"\trationale: {rationale}"
        )

        # Detect and break out of repeated-action loops
        if looks_stuck(action, args):
            print("\tdetected being stuck in loop...")
            state.last_observation = (
                "Loop detected: choose a different action."
            )
            # Skip execution; let the planner retry with updated context
            continue

        # Execute the chosen action
        ok, obs, payload, ms = execute_action(action, args)
        print(f"\t\ttool payload: {payload}")

        # Record telemetry for this step
        state.steps_taken += 1
        state.tool_trace.append(StepRecord(
            action=action,
            args=args,
            ok=ok,
            latency_ms=ms,
            info=payload
        ))

        # Feed observation back to planner for the next iteration
        state.last_observation = obs

        # Compress new evidence into the rolling summary
        update_summary(state, f"{action}({args}) -> {obs}")

        # Produce final answer and exit when planner signals 'answer'
        if action == "answer" and ok:
            final = synthesize_answer(state)
            state.done = True
            return final

        # On tool failure the planner sees the observation and can pivot
    print(within_budget(state), state.done)
    # Only reached if all budgets are exhausted with no answer produced
    return "Stopped: budget exhausted or no progress."


# Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    # Accept a single positional query argument from the command line.
    # Example usage:
    #   python agentic_controller.py "What is the weather in Baton Rouge?"
    #   python agentic_controller.py "Where is the ISS right now?"
    #   python agentic_controller.py "My name is Daniel, please remember it."
    parser = argparse.ArgumentParser(description="Agentic Controller")
    parser.add_argument("query", type=str, help="Your query to the agent")
    parsed = parser.parse_args()

    print("\n--- Running Agent ---\n")
    answer = run_agent(parsed.query)
    print("\n--- Final Answer ---\n")
    print(answer)