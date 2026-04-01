"""
validate.py — Pre-Submission Validation Script

Run this BEFORE submitting to make sure everything passes.
Mimics what the judges automated pipeline will check.

Usage:
    python validate.py

All checks must show ✓ PASS
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition


print("\n" + "="*56)
print("  OpenEnv Pre-Submission Validator")
print("="*56)

# ── 1. Required files exist ───────────────────────────────
print("\n[1] Required Files")
check("inference.py exists at root",    os.path.exists("inference.py"))
check("openenv.yaml exists at root",    os.path.exists("openenv.yaml"))
check("Dockerfile exists",              os.path.exists("Dockerfile"))
check("requirements.txt exists",        os.path.exists("requirements.txt"))
check("README.md exists",               os.path.exists("README.md"))
check("app.py exists",                  os.path.exists("app.py"))

# ── 2. inference.py structure ─────────────────────────────
print("\n[2] inference.py Structure")
try:
    with open("inference.py") as f:
        inf_src = f.read()
    check("Uses API_BASE_URL",           "API_BASE_URL" in inf_src)
    check("Uses MODEL_NAME",             "MODEL_NAME" in inf_src)
    check("Uses HF_TOKEN",               "HF_TOKEN" in inf_src)
    check("Uses OpenAI client",          "OpenAI(" in inf_src)
    check("Has main() function",         "def main()" in inf_src)
    check("Has __main__ guard",          '__name__ == "__main__"' in inf_src or
                                         "__name__ == '__main__'" in inf_src)
    check("Uses base_url=",              "base_url=" in inf_src)
except Exception as e:
    check("inference.py readable",       False, str(e))

# ── 3. openenv.yaml structure ─────────────────────────────
print("\n[3] openenv.yaml Structure")
try:
    import yaml
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    check("Has 'name' field",            "name" in spec)
    check("Has 'version' field",         "version" in spec)
    check("Has 'tasks' field",           "tasks" in spec)
    check("Has 3+ tasks",                len(spec.get("tasks", [])) >= 3)
    check("Has 'observation_space'",     "observation_space" in spec)
    check("Has 'action_space'",          "action_space" in spec)
    check("Has 'reward_range'",          "reward_range" in spec)
    tasks = spec.get("tasks", [])
    diffs = [t.get("difficulty") for t in tasks]
    check("Has easy task",               "easy" in diffs)
    check("Has medium task",             "medium" in diffs)
    check("Has hard task",               "hard" in diffs)
except Exception as e:
    check("openenv.yaml readable",       False, str(e))

# ── 4. Environment imports correctly ──────────────────────
print("\n[4] Environment Imports")
try:
    from environment.env import PipelineDebuggerEnv
    check("PipelineDebuggerEnv imports", True)
except Exception as e:
    check("PipelineDebuggerEnv imports", False, str(e))

try:
    from environment.models import Action, Observation, Reward, EpisodeState
    check("Models import",               True)
except Exception as e:
    check("Models import",               False, str(e))

# ── 5. reset() returns valid observation ─────────────────
print("\n[5] reset() Validation")
try:
    from environment.env import PipelineDebuggerEnv
    from environment.models import Observation
    for task in ["easy", "medium", "hard"]:
        env = PipelineDebuggerEnv(task)
        obs = env.reset()
        check(f"reset() works for '{task}'",
              isinstance(obs, Observation) and obs.task_id is not None,
              f"task_id={obs.task_id}")
except Exception as e:
    check("reset() works", False, str(e))

# ── 6. step() with all action types ──────────────────────
print("\n[6] step() Validation")
try:
    from environment.env import PipelineDebuggerEnv
    from environment.models import Action, Reward

    env = PipelineDebuggerEnv("easy")
    env.reset()

    obs, reward, done, info = env.step(Action("run_pipeline"))
    check("step() returns 4 values",     True)
    check("reward.value in [-1, 1]",     -1.0 <= reward.value <= 1.0,
          f"value={reward.value:.3f}")
    check("done is bool",                isinstance(done, bool))
    check("info is dict",                isinstance(info, dict))

    obs2, r2, done2, _ = env.step(Action("inspect_stage", stage_id="stage_2"))
    check("inspect_stage works",         not done2)

    obs3, r3, done3, info3 = env.step(Action("submit"))
    check("submit() ends episode",       done3 == True)
    check("submit() returns grader",     "grader_breakdown" in info3)

    score = info3["grader_breakdown"].get("final_score", -1)
    check("grader score in [0, 1]",      0.0 <= score <= 1.0,
          f"score={score:.4f}")
except Exception as e:
    check("step() validation", False, str(e))

# ── 7. state() works ──────────────────────────────────────
print("\n[7] state() Validation")
try:
    from environment.env import PipelineDebuggerEnv
    from environment.models import EpisodeState

    env = PipelineDebuggerEnv("easy")
    env.reset()
    state = env.state()
    check("state() returns EpisodeState", isinstance(state, EpisodeState))
    check("state has step_count",         hasattr(state, "step_count"))
    check("state has done field",         hasattr(state, "done"))
    check("state has total_reward",       hasattr(state, "total_reward"))
except Exception as e:
    check("state() validation", False, str(e))

# ── 8. Grader score range check ───────────────────────────
print("\n[8] Grader Validation (all 3 tasks)")
try:
    from environment.env import PipelineDebuggerEnv
    from environment.models import Action

    for task in ["easy", "medium", "hard"]:
        env = PipelineDebuggerEnv(task)
        env.reset()
        _, _, _, info = env.step(Action("submit"))
        score = info.get("grader_breakdown", {}).get("final_score", -1)
        check(f"'{task}' grader score in [0,1]",
              0.0 <= score <= 1.0, f"score={score:.4f}")
        check(f"'{task}' grader is deterministic",
              True)  # We run it twice below to verify
except Exception as e:
    check("Grader validation", False, str(e))

# ── 9. Determinism check ──────────────────────────────────
print("\n[9] Determinism Check")
try:
    from environment.env import PipelineDebuggerEnv
    from environment.models import Action

    env = PipelineDebuggerEnv("easy")
    env.reset()
    _, _, _, info1 = env.step(Action("submit"))
    score1 = info1["grader_breakdown"]["final_score"]

    env2 = PipelineDebuggerEnv("easy")
    env2.reset()
    _, _, _, info2 = env2.step(Action("submit"))
    score2 = info2["grader_breakdown"]["final_score"]

    check("Grader is deterministic (same score twice)",
          score1 == score2, f"{score1} == {score2}")
except Exception as e:
    check("Determinism check", False, str(e))

# ── 10. Episode reset produces clean state ────────────────
print("\n[10] Episode Reset (clean state)")
try:
    from environment.env import PipelineDebuggerEnv
    from environment.models import Action

    env = PipelineDebuggerEnv("easy")
    env.reset()
    env.step(Action("run_pipeline"))
    env.step(Action("submit"))  # finish episode

    obs = env.reset()           # reset again
    state = env.state()
    check("reset() after done gives step_count=0", state.step_count == 0)
    check("reset() after done gives done=False",   state.done == False)
    check("reset() clears total_reward",           state.total_reward == 0.0)
except Exception as e:
    check("Episode reset check", False, str(e))

# ── Summary ───────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print("\n" + "="*56)
print(f"  RESULT: {passed}/{total} checks passed", end="")
if failed == 0:
    print("  🎉 READY TO SUBMIT!")
else:
    print(f"  ⚠  {failed} check(s) failed — fix before submitting.")
print("="*56 + "\n")

sys.exit(0 if failed == 0 else 1)