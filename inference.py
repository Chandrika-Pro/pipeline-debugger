"""
inference.py — Pipeline Debugger Baseline Agent
===================================
MANDATORY:
- API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
- MODEL_NAME     The model identifier to use for inference
- HF_TOKEN       Your Hugging Face / API key

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    export HF_TOKEN="hf_your_token_here"
    python inference.py
"""

import os
import sys
import json
import textwrap
import time
from typing import List, Dict, Any, Optional

from openai import OpenAI

# ── Bring environment into path ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment.env import PipelineDebuggerEnv
from environment.models import Action

# ── Required env variables (as per spec) ─────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

# ── Inference config ──────────────────────────────────────
MAX_STEPS_PER_TASK = {"easy": 12, "medium": 18, "hard": 25}
TEMPERATURE        = 0.1
MAX_TOKENS         = 400
TASKS              = ["easy", "medium", "hard"]

# ── System prompt ─────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data engineer. You are debugging a broken data pipeline.

You will see:
- A description of the broken pipeline
- Each pipeline stage with its Python/pandas code
- Sample input data
- Current (broken) output
- Expected (correct) output
- Recent logs and error messages

Your goal: find and fix all bugs so the output matches expected.

You interact by outputting ONLY a valid JSON action. Nothing else. No explanation.

AVAILABLE ACTIONS:

Inspect a stage to see its full details:
{"action_type": "inspect_stage", "stage_id": "stage_1"}

Fix a stage's transformation code:
{"action_type": "fix_transform", "stage_id": "stage_2", "new_code": "df['price'] = df['price'].astype(float)\\nresult = df", "reasoning": "price should be float"}

Fix a column type (shortcut):
{"action_type": "fix_schema", "stage_id": "stage_2", "column_name": "price", "new_type": "float"}

Fix a filter condition:
{"action_type": "fix_filter", "stage_id": "stage_3", "new_condition": "df['date'] > pd.Timestamp('2024-02-01')"}

Run the pipeline to check current state:
{"action_type": "run_pipeline"}

Submit your final fix for grading:
{"action_type": "submit", "reasoning": "All bugs fixed"}

STRATEGY:
1. First run_pipeline to see what errors exist
2. Inspect suspicious stages
3. Fix bugs (earliest stage first — order matters!)
4. Run pipeline again to verify fix
5. Submit when output looks correct

Output ONLY valid JSON. No markdown, no explanation.
""").strip()


def build_prompt(obs) -> str:
    """Convert an Observation into a text prompt for the LLM."""
    stages_text = ""
    for s in obs.pipeline_stages:
        stages_text += f"\n[{s.stage_id}] type={s.stage_type} | status={s.status}"
        if s.error_message:
            stages_text += f" | ERROR: {s.error_message}"
        stages_text += f"\nDescription: {s.description}"
        stages_text += f"\nCode:\n{s.code}\n"

    logs_text = "\n".join(obs.logs[-6:]) if obs.logs else "No logs yet."
    history_text = (
        "\n".join(obs.actions_taken[-4:])
        if obs.actions_taken else "No actions yet."
    )

    return textwrap.dedent(f"""
TASK: {obs.task_description}

--- PIPELINE STAGES ---
{stages_text}

--- INPUT DATA (first 3 rows) ---
{json.dumps(obs.input_data_sample[:3], indent=2, default=str)}

--- CURRENT OUTPUT (broken) ---
{json.dumps(obs.current_output_sample[:3], indent=2, default=str)}

--- EXPECTED OUTPUT (correct) ---
{json.dumps(obs.expected_output_sample[:3], indent=2, default=str)}

--- EXPECTED SCHEMA ---
{json.dumps(obs.expected_schema, indent=2)}

--- RECENT LOGS ---
{logs_text}

--- ACTIONS TAKEN SO FAR ---
{history_text}

Steps remaining: {obs.steps_remaining}
Current score:   {obs.current_score:.3f}

Output ONLY a valid JSON action.
""").strip()


def parse_action(response_text: str) -> Optional[Action]:
    """Parse LLM response into an Action. Returns None if unparseable."""
    if not response_text:
        return None

    text = response_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        data = json.loads(text[start:end])
        return Action(**data)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def run_task(client: OpenAI, task_name: str) -> Dict[str, Any]:
    """Run one task with the LLM agent."""

    # ── START log (required structured format) ────────────
    print(f"START task={task_name} model={MODEL_NAME}")

    env      = PipelineDebuggerEnv(task_name)
    obs      = env.reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    max_steps        = MAX_STEPS_PER_TASK[task_name]
    total_reward     = 0.0
    final_score      = 0.0
    steps_used       = 0
    done             = False
    grader_breakdown = {}

    while not done and steps_used < max_steps:

        user_prompt = build_prompt(obs)
        messages.append({"role": "user", "content": user_prompt})

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"STEP {steps_used+1} action=run_pipeline reward=0.000 error={exc}")
            response_text = '{"action_type": "run_pipeline"}'

        messages.append({"role": "assistant", "content": response_text})

        action = parse_action(response_text)
        if action is None:
            action = Action(action_type="run_pipeline")

        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps_used   += 1

        # ── STEP log (required structured format) ─────────
        action_str = action.action_type
        if action.stage_id:
            action_str += f":{action.stage_id}"
        print(f"STEP {steps_used} action={action_str} reward={reward.value:.3f} score={obs.current_score:.3f}")

        if done and "grader_breakdown" in info:
            grader_breakdown = info["grader_breakdown"]
            final_score      = grader_breakdown.get("final_score", 0.0)

    # Force submit if max steps reached
    if not done:
        obs, reward, done, info = env.step(Action(action_type="submit"))
        steps_used += 1
        if "grader_breakdown" in info:
            grader_breakdown = info["grader_breakdown"]
            final_score      = grader_breakdown.get("final_score", 0.0)
        print(f"STEP {steps_used} action=submit reward={reward.value:.3f} score={final_score:.3f}")

    # ── END log (required structured format) ──────────────
    print(f"END task={task_name} final_score={final_score:.4f} steps={steps_used}")

    return {
        "task":             task_name,
        "final_score":      round(final_score, 4),
        "steps_used":       steps_used,
        "total_reward":     round(total_reward, 4),
        "grader_breakdown": grader_breakdown,
    }


def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN environment variable.")
        sys.exit(1)

    print(f"Pipeline Debugger — Baseline Inference")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  Tasks        : {TASKS}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results    = []
    start_time = time.time()

    for task_name in TASKS:
        result = run_task(client, task_name)
        results.append(result)

        elapsed = time.time() - start_time
        if elapsed > 18 * 60:
            print(f"WARNING: Approaching time limit ({elapsed/60:.1f}min). Stopping.")
            break

    # ── Summary ───────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*56}")
    print(f"  BASELINE RESULTS  |  model: {MODEL_NAME}")
    print(f"{'='*56}")
    print(f"  {'Task':<10} {'Score':>8}   {'Steps':>6}")
    print(f"  {'-'*30}")
    for r in results:
        print(f"  {r['task']:<10} {r['final_score']:>8.4f}   {r['steps_used']:>6}")

    avg = sum(r["final_score"] for r in results) / len(results) if results else 0.0
    print(f"  {'-'*30}")
    print(f"  {'AVERAGE':<10} {avg:>8.4f}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"{'='*56}\n")

    # Save results
    output = {
        "model":         MODEL_NAME,
        "api_base_url":  API_BASE_URL,
        "tasks":         TASKS,
        "results":       results,
        "average_score": round(avg, 4),
        "elapsed_sec":   round(elapsed, 1),
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()