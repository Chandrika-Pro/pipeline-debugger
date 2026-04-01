"""
tests/test_env.py — The Test Suite

Think of these like practice exam papers.
Before submitting the real exam (hackathon), we run these
tests to make sure everything works correctly.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.env import PipelineDebuggerEnv
from environment.models import Action, Observation, Reward, EpisodeState


# ──────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────

def make_env(task="easy"):
    env = PipelineDebuggerEnv(task)
    obs = env.reset()
    return env, obs


# ──────────────────────────────────────────────────────────
# TEST 1: reset() works and returns a valid Observation
# ──────────────────────────────────────────────────────────

def test_reset_returns_observation():
    print("\n[TEST 1] reset() returns valid Observation...")
    env, obs = make_env("easy")

    assert isinstance(obs, Observation), "reset() must return Observation"
    assert obs.task_id == "easy_type_mismatch"
    assert len(obs.pipeline_stages) == 4, "Easy task should have 4 stages"
    assert len(obs.expected_output_sample) > 0, "Expected output sample must exist"
    assert obs.steps_remaining == 15, "Easy task has 15 max steps"

    print("  ✓ PASSED")


# ──────────────────────────────────────────────────────────
# TEST 2: state() returns EpisodeState after reset
# ──────────────────────────────────────────────────────────

def test_state_after_reset():
    print("\n[TEST 2] state() returns EpisodeState after reset...")
    env, _ = make_env("easy")
    state = env.state()

    assert isinstance(state, EpisodeState)
    assert state.step_count == 0
    assert state.done == False
    assert state.total_reward == 0.0

    print("  ✓ PASSED")


# ──────────────────────────────────────────────────────────
# TEST 3: step() with inspect_stage works
# ──────────────────────────────────────────────────────────

def test_inspect_stage():
    print("\n[TEST 3] inspect_stage action works...")
    env, _ = make_env("easy")
    obs, reward, done, info = env.step(Action("inspect_stage", stage_id="stage_2"))

    assert isinstance(obs, Observation)
    assert isinstance(reward, Reward)
    assert done == False
    assert reward.value >= -0.1, "Inspecting should not give a big penalty"

    state = env.state()
    assert state.step_count == 1

    print("  ✓ PASSED")


# ──────────────────────────────────────────────────────────
# TEST 4: run_pipeline gives negative reward when broken
# ──────────────────────────────────────────────────────────

def test_run_broken_pipeline():
    print("\n[TEST 4] run_pipeline on broken pipeline gives penalty...")
    env, _ = make_env("easy")
    obs, reward, done, info = env.step(Action("run_pipeline"))

    # Easy task pipeline crashes on broken state (int cast on float string)
    assert reward.value <= 0.05, f"Broken pipeline should not give high reward, got {reward.value}"
    assert any("FAILED" in log or "✗" in log for log in obs.logs), \
        "Logs should show pipeline failure"

    print("  ✓ PASSED")


# ──────────────────────────────────────────────────────────
# TEST 5: fix_transform fixes the easy bug
# ──────────────────────────────────────────────────────────

def test_fix_easy_bug():
    print("\n[TEST 5] fix_transform fixes the easy type cast bug...")
    env, _ = make_env("easy")

    fix_code = """
df['price'] = df['price'].astype(float)
df['quantity'] = df['quantity'].astype(int)
result = df
"""
    obs, reward, done, info = env.step(
        Action("fix_transform", stage_id="stage_2", new_code=fix_code)
    )

    assert reward.value > 0, f"Fixing a bug should give positive reward, got {reward.value}"
    assert not any("FAILED" in log for log in obs.logs), \
        "Pipeline should run clean after fix"

    print(f"  ✓ PASSED — reward after fix: {reward.value:.3f}")


# ──────────────────────────────────────────────────────────
# TEST 6: submit after correct fix scores high
# ──────────────────────────────────────────────────────────

def test_submit_easy_correct_fix():
    print("\n[TEST 6] submit after correct fix scores >= 0.8...")
    env, _ = make_env("easy")

    fix_code = """
df['price'] = df['price'].astype(float)
df['quantity'] = df['quantity'].astype(int)
result = df
"""
    env.step(Action("fix_transform", stage_id="stage_2", new_code=fix_code))
    obs, reward, done, info = env.step(Action("submit"))

    assert done == True, "submit() must end the episode"
    assert reward.value >= 0.8, f"Correct fix should score >= 0.8, got {reward.value:.4f}"
    assert "grader_breakdown" in info

    print(f"  ✓ PASSED — final score: {reward.value:.4f}")


# ──────────────────────────────────────────────────────────
# TEST 7: submit without fixing scores low
# ──────────────────────────────────────────────────────────

def test_submit_without_fixing():
    print("\n[TEST 7] submit without fixing scores low...")
    env, _ = make_env("easy")
    obs, reward, done, info = env.step(Action("submit"))

    assert done == True
    assert reward.value < 0.5, f"Unfixed pipeline should score < 0.5, got {reward.value:.4f}"

    print(f"  ✓ PASSED — score without fix: {reward.value:.4f}")


# ──────────────────────────────────────────────────────────
# TEST 8: Cannot step after episode is done
# ──────────────────────────────────────────────────────────

def test_cannot_step_after_done():
    print("\n[TEST 8] Cannot step after episode is done...")
    env, _ = make_env("easy")
    env.step(Action("submit"))  # End episode

    try:
        env.step(Action("run_pipeline"))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "done" in str(e).lower()
        print("  ✓ PASSED — RuntimeError raised as expected")


# ──────────────────────────────────────────────────────────
# TEST 9: reset() after done starts fresh
# ──────────────────────────────────────────────────────────

def test_reset_after_done():
    print("\n[TEST 9] reset() after done starts a fresh episode...")
    env, _ = make_env("easy")
    env.step(Action("submit"))  # End episode

    obs = env.reset()  # Fresh start
    state = env.state()

    assert state.step_count == 0
    assert state.done == False
    assert state.total_reward == 0.0
    assert obs.steps_remaining == 15

    print("  ✓ PASSED")


# ──────────────────────────────────────────────────────────
# TEST 10: Medium task loads correctly
# ──────────────────────────────────────────────────────────

def test_medium_task_loads():
    print("\n[TEST 10] Medium task loads and pipeline runs (no crash)...")
    env, obs = make_env("medium")

    assert obs.task_id == "medium_silent_filter_bug"
    assert obs.steps_remaining == 20

    # Medium task should NOT crash (silent bug)
    obs2, reward, done, info = env.step(Action("run_pipeline"))
    # Medium task pipeline runs without errors — silent bug
    assert reward.value >= -0.05, "Medium task shouldn't crash immediately"

    print("  ✓ PASSED")


# ──────────────────────────────────────────────────────────
# TEST 11: Hard task loads with 5 stages
# ──────────────────────────────────────────────────────────

def test_hard_task_loads():
    print("\n[TEST 11] Hard task loads with 5 stages...")
    env, obs = make_env("hard")

    assert obs.task_id == "hard_cascade_bugs"
    assert len(obs.pipeline_stages) == 5, f"Hard task should have 5 stages, got {len(obs.pipeline_stages)}"
    assert obs.steps_remaining == 30

    print("  ✓ PASSED")


# ──────────────────────────────────────────────────────────
# TEST 12: Invalid task name raises ValueError
# ──────────────────────────────────────────────────────────

def test_invalid_task_raises():
    print("\n[TEST 12] Invalid task name raises ValueError...")
    try:
        env = PipelineDebuggerEnv("impossible_task")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ PASSED — ValueError: {e}")


# ──────────────────────────────────────────────────────────
# TEST 13: Invalid action type raises ValueError
# ──────────────────────────────────────────────────────────

def test_invalid_action_raises():
    print("\n[TEST 13] Invalid action type raises ValueError...")
    try:
        action = Action("fly_to_the_moon")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ PASSED — ValueError: {e}")


# ──────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_reset_returns_observation,
        test_state_after_reset,
        test_inspect_stage,
        test_run_broken_pipeline,
        test_fix_easy_bug,
        test_submit_easy_correct_fix,
        test_submit_without_fixing,
        test_cannot_step_after_done,
        test_reset_after_done,
        test_medium_task_loads,
        test_hard_task_loads,
        test_invalid_task_raises,
        test_invalid_action_raises,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*40}")

    if failed > 0:
        sys.exit(1)