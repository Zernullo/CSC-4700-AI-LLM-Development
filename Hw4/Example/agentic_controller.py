"""
agentic_controller.py

This is just a  reference implementation of a minimal agentic controller loop with planning.

What is in this file:
- Tool catalog with strict JSON Schemas (prevents tool/field hallucination)
- Argument validation + one-shot LLM repair when validation fails
- Budgets: max steps, tokens, and cost with simple accounting
- Loop detection to stop repeated ineffective actions
- Rolling history summarization to keep context small
- Planner that chooses the next action (tool or 'answer')
- Executor stub that simulates two tools (replace with your backends)
- Final synthesis step that composes the final answer

Notes on LLM Provider:
- This version uses OpenAI's SDK for planning, repair, summarization, and synthesis.
- To swap providers, replace client calls in the following functions:
  - repair_args_with_llm()
  - update_summary()
  - plan_next_action()
  - synthesize_answer()

Run:
  python agentic_controller.py

Modified by Samuel Hildebrand, 2026-02
"""

# Imports & Setup ------------------------------------------------------------------------------------------------------

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
import random

# Load environment variables
load_dotenv('../../.env')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool Catalog ---------------------------------------------------------------------------------------------------------
# the tool catalog provides precise JSON Schemas for arguments. This helps to prevent the model from inventing fields or
# tools, and helps validation & auto-repair.

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # Tool 1: a weather tool
    # STUDENT_COMPLETE --> You need to replace this with the correct one for the real weather tool call
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
    # Tool 2: knowledge-base search tool
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
    # STUDENT_COMPLETE --> Document the read_mem tool here for the model here. You can find the code for it
    # in the execute_action function. 
    "read_mem": {}, # Finish implementing the JSON Schema for the read_mem tool. It should not require any arguments.
    # STUDENT_COMPLETE --> Document the write_mem tool here for the model here. You can find the code for it in the execute_action function.
    "write_mem": {
        "type": "object",
        "description": "", # Add a Description
        "properties": {
            "content": {} # Finish implementing content
        },
        "required": ["content"],
        "additionalProperties": False
    },

    # STUDENT_COMPLETE --> You need to add a new tool schema for your custom tool

}

# Optional hints: rough latency/cost so planner can reason about budgets. I recommend replacing the default values
# with estimates that are accurate based on measurements.
TOOL_HINTS: Dict[str, Dict[str, Any]] = {
    "weather.get_current": {"avg_ms": 400, "avg_tokens": 50},
    "kb.search":           {"avg_ms": 120, "avg_tokens": 30},
    "read_mem": {"avg_ms": 5, "avg_tokens": 200},
    "write_mem": {"avg_ms": 5, "avg_tokens": 50},
}

# Controller State -----------------------------------------------------------------------------------------------------
@dataclass
class StepRecord:
    """Telemetry for each executed step (action)."""
    action: str                   # tool name or 'answer'
    args: Dict[str, Any]          # arguments supplied
    ok: bool                      # success flag
    latency_ms: int               # latency in milliseconds
    info: Dict[str, Any] = field(default_factory=dict)  # normalized payload

@dataclass
class ControllerState:
    """Mutable task state carried through the controller loop."""
    goal: str                     # user task/goal
    history_summary: str = ""     # compact running summary (LLM-generated)
    tool_trace: List[StepRecord] = field(default_factory=list)
    tokens_used: int = 0          # simple token accounting
    cost_cents: float = 0.0       # simple cost accounting
    steps_taken: int = 0          # how many actions executed
    last_observation: str = ""    # short feedback string from last step
    done: bool = False            # termination flag


# Budgets & Accounting -------------------------------------------------------------------------------------------------
# Hard ceilings to avoid runaway cost
MAX_STEPS = 8
MAX_TOKENS = 20_000
MAX_COST_CENTS = 75.0


def within_budget(s: ControllerState) -> bool:
    """
    Check hard ceilings for steps, tokens, and cost.

    :param s: instance of ControllerState
    :return: True if still within budget, false if over-budget
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
    # gpt-5-mini is $0.25/million token
    s.cost_cents += total * 0.25/1E4


# Loop Detection -------------------------------------------------------------------------------------------------------
# Detect repeated (action, args) to avoid "stuck" ReAct oscillations.
LAST_ACTIONS = deque(maxlen=3)


def fingerprint_action(action: str, args: Dict[str, Any]) -> str:
    """
    Hash the tool call pair (action,args) to compare recent moves.

    :param action: the action the model selected
    :param args: the arguments the model selected for the action
    :return: A sha256 hash
    """
    blob = json.dumps({"a": action, "x": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def looks_stuck(action: str, args: Dict[str, Any]) -> bool:
    """
    Return True if the last N actions are identical (loop).

    :param action: the action the model selected
    :param args: the arguments the model selected for the action
    :return: True if this action is the same as the N actions, False otherwise
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
    Validate args against the JSON Schema for the given tool. Return (ok, error_message). Error is concise for LLM
    repair prompt.

    :param tool_name: the name of the tool that the model selected
    :param args: the arguments the model selected for that tool
    :return: (True, None) if validates, (False, error message) if not
    """
    schema = TOOL_SCHEMAS[tool_name]
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(args), key=lambda e: e.path)
    if errors:
        e = errors[0]
        path = "/".join([str(p) for p in e.path]) or "(root)"
        return False, f"Invalid arguments at {path}: {e.message}"
    return True, None


def repair_args_with_llm(tool_name: str, bad_args: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """
    Ask the LLM to fix only the invalid parts to satisfy the JSON Schema.
    We enforce JSON-only output and re-validate after repair.

    :param tool_name: name of the selected tool
    :param bad_args: dictionary of bad arguments provided by the model
    :param error_msg: the error message provided by the validator
    :return: corrected (hopefully) arguments
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
        model="gpt-5-mini",
        response_format={"type": "json_object"},  # force JSON
        messages=[
            {"role": "developer", "content": dev},
            {"role": "user", "content": user}
        ]
    )
    return json.loads(resp.choices[0].message.content)


# History Summarization ------------------------------------------------------------------------------------------------

def update_summary(state: ControllerState, new_evidence: str) -> None:
    """
    Compress the prior summary + new evidence into a short rolling memory.
    Keeps context small but preserves key facts and decisions.

    :param state: instance of ControllerState
    :param new_evidence: this would be the response from the tool call
    :return:
    """
    sys = "Compress facts and decisions into ≤120 tokens. Keep IDs and key numbers. Do not include anything that is " \
          "unnecessary, only things that are strictly useful for the goal."
    user = json.dumps({
        "prior_summary": state.history_summary,
        "new_evidence": new_evidence
    })
    resp = client.chat.completions.create(
        model="gpt-5-mini",
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
    Ask the LLM to pick ONE next action:
      - a known tool from TOOL_SCHEMAS with arguments, OR
      - the literal string 'answer' when it can synthesize the final answer.

    :param state: instance of ControllerState
    :return: (action, args, rationale)
    """
    # Pass the schema to the model. We also pass tool latency/token count for budget control and example to help the
    # model choose.
    tool_specs = []
    for name, schema in TOOL_SCHEMAS.items():
        spec = {
            "name": name,
            "schema": schema,  # including the full JSON Schema
            "budget_hint": {
                "avg_ms": TOOL_HINTS[name]["avg_ms"],
                "avg_tokens": TOOL_HINTS[name]["avg_tokens"],
            },
            # (Optional Few-shot prompting approach) keep examples tiny and schema-compliant
            # STUDENT_COMPLETE --> You may need to change this to be in line with your custom weather tool implementation
            "examples": {
                "weather.get_current": [
                    {"city": "Paris", "units": "metric"},
                    {"city": "New York"}  # units defaults via schema
                ],
                "kb.search": [
                    {"query": "VPN policy for contractors", "k": 3}
                ]
            }.get(name, [])
        }
        tool_specs.append(spec)

    dev = (
        "You are a planner. Choose ONE next action toward the goal. Do not call actions towards "
        "information already contained in the history summary provided below.\n"
        "Use ONLY tools from `tool_catalog` OR choose 'answer' if you can respond now.\n"
        "You can only answer with information provided by the tools."
        "When using a tool, produce arguments that VALIDATE against its JSON Schema.\n"
        "Allowed output format (JSON only):\n"
        '{"action":"<tool_name|answer>","args":{...}, "rationale":"<brief reason>"}'
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
        model="gpt-5-mini",
        response_format={"type": "json_object"},  # we could use strict mode if we wanted to
        messages=[{"role": "developer", "content": dev},
                  {"role": "user", "content": user}]
    )
    obj = json.loads(resp.choices[0].message.content)
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)
    return obj["action"], obj.get("args", {}), obj.get("rationale", "")


# Executor -------------------------------------------------------------------------------------------------------------

def execute_action(action: str, args: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any], int]:
    """
    Execute the selected action with validation, repair, retries, and error handling.

    Replace the stubbed tool bodies with your real backends (APIs, DBs, etc.). Right now, they are just dummie entries.

    :param action: tool selected by the model
    :param args: arguments selected by the model
    :return: (ok, observation_text, normalized_payload, latency_ms)
    """

    t0 = time.time()

    # 'answer' is a "virtual tool" signaling that we should synthesize the final answer.
    if action == "answer":
        obs = "Ready to synthesize final answer from working memory and evidence."
        return True, obs, {}, int((time.time() - t0) * 1000)

    # Guard: only call known tools from the catalog.
    if action not in TOOL_SCHEMAS:
        return False, f"Unknown tool: {action}", {}, int((time.time() - t0) * 1000)

    # 1) Validate arguments against schema.
    ok, msg = validate_args(action, args)
    if not ok:
        # 2) One-shot repair via LLM; re-validate.
        fixed = repair_args_with_llm(action, args, msg)
        ok2, msg2 = validate_args(action, fixed)
        if not ok2:
            return False, f"Arg repair failed: {msg2}", {}, int((time.time() - t0) * 1000)
        args = fixed

    # 3) Execute the tool with basic retry on transient failures (e.g., timeouts).
    try:
        # Replace these with your real integrations
        if action == "weather.get_current":  # STUDENT_COMPLETE --> make this an actual weather API call
            # Simulate an external API call to a weather service
            # (In production, you would call your real weather API here.)
            payload = {
                "city": args["city"],
                "units": args.get("units", "metric"),
                "temp": 18.2,
                "conditions": "clear"
            }
            obs = f"Weather in {payload['city']}: {payload['temp']}° ({payload['conditions']})"
            return True, obs, payload, int((time.time() - t0) * 1000)

        elif action == "kb.search":  # STUDENT_COMPLETE --> make this a vector search over a Chroma database
            # Simulate a KB or vector database search
            k = int(args.get("k", 5))
            results = [
                {"doc_id": "kb-12", "snippet": "VPN and MFA are required for remote access."},
                {"doc_id": "kb-31", "snippet": "Employees must complete security training annually."}
            ][:k]
            obs = f"Retrieved {len(results)} snippets"
            return True, obs, {"results": results}, int((time.time() - t0) * 1000)

        elif action == "read_mem":
            memory_path = os.path.join(os.path.dirname(__file__), "Memory.md")

            try:
                # Ensure file exists
                if not os.path.exists(memory_path):
                    with open(memory_path, "w", encoding="utf-8") as f:
                        f.write("")

                # Read memory contents
                with open(memory_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Put content in the observation so it enters working memory.
                obs = f"Memory contents:\n{content}"

                payload = {"content": content}

                return True, obs, payload, int((time.time() - t0) * 1000)

            except Exception as e:
                return False, f"Memory read error: {e}", {}, int((time.time() - t0) * 1000)
            
        elif action == "write_mem":
            memory_path = os.path.join(os.path.dirname(__file__), "Memory.md")

            try:
                content = args["content"]

                with open(memory_path, "a", encoding="utf-8") as f:
                    f.write(content + "\n")

                obs = f"Appended {len(content)} characters to memory."
                payload = {"written": len(content)}

                return True, obs, payload, int((time.time() - t0) * 1000)

            except Exception as e:
                return False, f"Memory write error: {e}", {}, int((time.time() - t0) * 1000)

        else:
            # Safety: no executor wired for this tool
            return False, f"No executor bound for tool: {action}", {}, int((time.time() - t0) * 1000)

    except Exception as e:
        # Non-transient or unexpected error
        return False, f"Tool error: {type(e).__name__}: {e}", {}, int((time.time() - t0) * 1000)


# Final Synthesis ------------------------------------------------------------------------------------------------------
def synthesize_answer(state: ControllerState) -> str:
    """
    Compose the final answer using the compact working summary accumulated in state.history_summary. The full raw trace
    can be logged elsewhere.

    :param state: instance of ControllerState
    :return: model's response
    """
    sys = "Your goal is to produce a final answer to a goal (likely a question) using only evidence provided in the " \
          "working summary."
    user = (
        f"Goal: {state.goal}\n\n"
        f"Working summary:\n{state.history_summary}\n\n"
        f"Produce the final answer in ≤ 200 tokens."
    )
    resp = client.chat.completions.create(
        model="gpt-5-mini",
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
        1) Build context (we keep a rolling summary in state)
        2) Plan next action (tool or 'answer')
        3) Loop detection guard
        4) Execute with validation/repair/retry
        5) Update summary and telemetry
        6) If 'answer', synthesize final output and stop

    :param goal:
    :return:
    """
    state = ControllerState(goal=goal)

    while within_budget(state) and not state.done:
        # Ask the planner to choose the next action
        action, args, rationale = plan_next_action(state)
        print(f"Action selected: {action}\n\targuments: {args}\n\trationale: {rationale}")

        # Prevent infinite ReAct loops by hashing last few actions
        if looks_stuck(action, args):
            print("\tdetected being stuck in loop...")
            state.last_observation = "Loop detected: revise plan with a different next action."
            # Do not increment steps or execute; let planner try again
            continue

        # Execute the chosen action (or 'answer' pseudo-tool)
        ok, obs, payload, ms = execute_action(action, args)
        print(f"\t\ttool payload: {payload}")

        # Record step telemetry
        state.steps_taken += 1
        state.tool_trace.append(StepRecord(
            action=action,
            args=args,
            ok=ok,
            latency_ms=ms,
            info=payload
        ))

        # Provide short observation back to planner for next turn
        state.last_observation = obs

        # Summarize new evidence into compact working memory
        update_summary(state, f"{action}({args}) -> {obs}")

        # If planner signaled 'answer', produce final answer and exit
        if action == "answer" and ok:
            final = synthesize_answer(state)
            state.done = True
            return final

        # If a tool failed, we do not crash; the planner sees the observation
        # and can pivot on the next iteration. The loop will also stop on budgets.
    print(within_budget(state), state.done)
    # If we exit naturally, budgets are exhausted or we never reached 'answer'
    return "Stopped: budget exhausted or no progress."


# Demo -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # STUDENT_COMPLETE --> you should use argparse here so that one can just ask a question like:
    #         python agentic_controller.py "Why were they trying to catch the whale in Moby Dick?"
    #         python agentic_controller.py "What is today's weather like in Baton Rouge?"
    #         python agentic_controller.py "INSERT SOME QUERY HERE RELEVANT TO YOUR CUSTOM TOOL"


    # Example end-to-end run:
    # The planner can choose to look up weather, search a KB, and then synthesize an answer.
    goal = "What's the current weather in Paris (metric) and do I need VPN for remote access?"
    print("\n--- Running Agent ---\n")
    answer = run_agent(goal)
    print("\n--- Final Answer ---\n")
    print(answer)

    # You could also print telemetry for inspection:
    # - steps taken, tokens used, cost, brief trace, etc.
