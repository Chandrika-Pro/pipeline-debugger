"""
env.py — The Game Engine

This is the heart of the environment.

Think of it like the code behind a board game:
- reset() → set up the board, deal cards, give first look to the player
- step()  → player makes a move, board updates, give points, check if game over
- state() → take a photo of the current board

The AI agent only talks to THIS file.
It never directly touches the pipeline or graders.
"""

from copy import deepcopy
from typing import Tuple, Dict, Any, Optional

import pandas as pd

from environment.models import (
    Observation, Action, Reward, EpisodeState, StageInfo
)
from environment import graders
from environment.tasks import task_easy, task_medium, task_hard


# Map task names to task creator functions
TASK_REGISTRY = {
    "easy":   task_easy.create_task,
    "medium": task_medium.create_task,
    "hard":   task_hard.create_task,
}


class PipelineDebuggerEnv:
    """
    The Data Pipeline Debugger Environment.

    An AI agent interacts with this class to:
    1. Get a broken data pipeline (reset)
    2. Inspect and fix stages (step)
    3. Score its fix (grader runs on submit)

    Compatible with the OpenEnv interface.
    """

    def __init__(self, task_name: str = "easy"):
        if task_name not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(TASK_REGISTRY.keys())}")

        self.task_name = task_name
        self._task_def = None        # loaded task definition
        self._pipeline = None        # current pipeline (agent modifies this)
        self._episode_state = None   # tracks step count, total reward, etc.
        self._last_output = None     # last DataFrame produced by run_pipeline

    # ──────────────────────────────────────────────────────────
    # reset() — Start a fresh episode
    # ──────────────────────────────────────────────────────────
    def reset(self) -> Observation:
        """
        Reset the environment to the start of a new episode.

        Like shuffling the deck and starting a new game.
        Returns the first Observation (what the agent sees at the start).
        """
        # Load a fresh copy of the task
        self._task_def = TASK_REGISTRY[self.task_name]()
        self._pipeline = self._task_def["pipeline"]

        # Run the pipeline once so agent sees the broken output immediately
        output, logs, _ = self._pipeline.run()
        self._last_output = output

        # Initialize episode state
        self._episode_state = EpisodeState(
            task_id=self._task_def["task_id"],
            step_count=0,
            done=False,
            total_reward=0.0,
            bugs_remaining=list(self._task_def["bugs"]),
            bugs_fixed=[],
        )

        obs = self._build_observation(logs)
        self._episode_state.current_observation = obs
        return obs

    # ──────────────────────────────────────────────────────────
    # step() — Agent takes an action
    # ──────────────────────────────────────────────────────────
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one agent action.

        Returns:
        - observation: what the agent sees now
        - reward: how many points it got for this action
        - done: is the episode over?
        - info: extra debug information
        """
        if self._episode_state is None:
            raise RuntimeError("Call reset() before step().")

        if self._episode_state.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new one.")

        self._episode_state.step_count += 1
        info = {"action": action.action_type, "success": False, "message": ""}

        # ── Handle each action type ────────────────────────
        action_log = f"Step {self._episode_state.step_count}: {action.action_type}"

        if action.action_type == "inspect_stage":
            obs, reward, done = self._handle_inspect(action, info)

        elif action.action_type == "fix_transform":
            obs, reward, done = self._handle_fix_transform(action, info)

        elif action.action_type == "fix_schema":
            obs, reward, done = self._handle_fix_schema(action, info)

        elif action.action_type == "fix_filter":
            obs, reward, done = self._handle_fix_filter(action, info)

        elif action.action_type == "reorder_stages":
            obs, reward, done = self._handle_reorder(action, info)

        elif action.action_type == "run_pipeline":
            obs, reward, done = self._handle_run_pipeline(action, info)

        elif action.action_type == "submit":
            obs, reward, done = self._handle_submit(action, info)

        else:
            # Unknown action
            reward = Reward(value=-0.05, reason="Unknown action type.")
            obs = self._build_observation([f"Unknown action: {action.action_type}"])
            done = False

        # Add action to history
        if obs.actions_taken is not None:
            obs.actions_taken.append(f"{action_log} → {info.get('message', '')}")

        # Update episode state
        self._episode_state.total_reward += reward.value
        self._episode_state.current_observation = obs
        self._episode_state.done = done

        # End episode if max steps reached
        steps_used = self._episode_state.step_count
        max_steps = self._task_def["max_steps"]
        if steps_used >= max_steps and not done:
            done = True
            self._episode_state.done = True
            info["message"] += " | Max steps reached."

        return obs, reward, done, info

    # ──────────────────────────────────────────────────────────
    # state() — Snapshot of current state
    # ──────────────────────────────────────────────────────────
    def state(self) -> EpisodeState:
        """Return a full snapshot of the current episode state."""
        if self._episode_state is None:
            raise RuntimeError("Call reset() first.")
        return self._episode_state

    # ──────────────────────────────────────────────────────────
    # Action Handlers (private)
    # ──────────────────────────────────────────────────────────

    def _handle_inspect(self, action: Action, info: dict):
        """inspect_stage → reveal the full code of a stage."""
        stage_info = self._pipeline.get_stage_info(action.stage_id)
        if stage_info is None:
            info["message"] = f"Stage '{action.stage_id}' not found."
            reward = Reward(value=-0.02, reason="Inspected a stage that doesn't exist.")
        else:
            info["success"] = True
            info["message"] = f"Inspected {action.stage_id}."
            # Small reward for useful inspection — no penalty
            reward = Reward(value=0.0, reason=f"Inspected stage {action.stage_id}. No points but no penalty.")

        logs = [f"Inspected stage: {action.stage_id}"]
        obs = self._build_observation(logs)
        return obs, reward, False

    def _handle_fix_transform(self, action: Action, info: dict):
        """fix_transform → replace a stage's code."""
        if not action.stage_id or not action.new_code:
            info["message"] = "fix_transform requires stage_id and new_code."
            reward = Reward(value=-0.05, reason="Invalid action parameters.")
            return self._build_observation([]), reward, False

        success = self._pipeline.fix_stage_code(action.stage_id, action.new_code)
        if not success:
            info["message"] = f"Stage '{action.stage_id}' not found."
            reward = Reward(value=-0.02, reason="Stage not found.")
        else:
            info["success"] = True
            info["message"] = f"Updated code in {action.stage_id}."
            reward = Reward(value=0.02, reason="Code updated. Run pipeline to see effect.")

        output, logs, run_success = self._pipeline.run()
        self._last_output = output if run_success else self._last_output

        # Give bonus if pipeline now runs clean
        if run_success and output is not None:
            reward.value += 0.05
            reward.reason += " Pipeline runs clean after fix."

        obs = self._build_observation(logs)
        return obs, reward, False

    def _handle_fix_schema(self, action: Action, info: dict):
        """fix_schema → add a type cast to a stage."""
        if not action.stage_id or not action.column_name or not action.new_type:
            reward = Reward(value=-0.05, reason="fix_schema requires stage_id, column_name, new_type.")
            return self._build_observation([]), reward, False

        success = self._pipeline.fix_schema(action.stage_id, action.column_name, action.new_type)
        if not success:
            reward = Reward(value=-0.02, reason="Invalid stage or type.")
            info["message"] = "Schema fix failed."
        else:
            info["success"] = True
            info["message"] = f"Schema fix applied to {action.stage_id}.{action.column_name}"
            reward = Reward(value=0.02, reason="Schema fix applied.")

        output, logs, run_success = self._pipeline.run()
        if run_success:
            self._last_output = output
            reward.value += 0.05
            reward.reason += " Pipeline now runs clean."

        obs = self._build_observation(logs)
        return obs, reward, False

    def _handle_fix_filter(self, action: Action, info: dict):
        """fix_filter → shorthand to fix a filter condition in stage code."""
        if not action.stage_id or not action.new_condition:
            reward = Reward(value=-0.05, reason="fix_filter requires stage_id and new_condition.")
            return self._build_observation([]), reward, False

        # Replace the stage code by appending the new condition
        stage_info = self._pipeline.get_stage_info(action.stage_id)
        if stage_info is None:
            reward = Reward(value=-0.02, reason="Stage not found.")
            return self._build_observation([]), reward, False

        new_code = f"result = df[{action.new_condition}].copy()"
        self._pipeline.fix_stage_code(action.stage_id, new_code)
        info["success"] = True
        info["message"] = f"Filter updated in {action.stage_id}."

        output, logs, run_success = self._pipeline.run()
        if run_success:
            self._last_output = output
            reward = Reward(value=0.07, reason="Filter fix applied, pipeline runs clean.")
        else:
            reward = Reward(value=0.01, reason="Filter fix applied but pipeline still has issues.")

        obs = self._build_observation(logs)
        return obs, reward, False

    def _handle_reorder(self, action: Action, info: dict):
        """reorder_stages → change the pipeline stage execution order."""
        if not action.new_order:
            reward = Reward(value=-0.05, reason="reorder_stages requires new_order list.")
            return self._build_observation([]), reward, False

        success = self._pipeline.reorder_stages(action.new_order)
        if not success:
            reward = Reward(value=-0.02, reason="Reorder failed — check stage IDs.")
        else:
            info["success"] = True
            reward = Reward(value=0.02, reason="Stages reordered.")

        output, logs, run_success = self._pipeline.run()
        if run_success:
            self._last_output = output
            reward.value += 0.05

        obs = self._build_observation(logs)
        return obs, reward, False

    def _handle_run_pipeline(self, action: Action, info: dict):
        """run_pipeline → execute pipeline and observe output."""
        output, logs, success = self._pipeline.run()

        if success:
            self._last_output = output
            reward = Reward(value=0.01, reason="Pipeline ran successfully.")
            info["success"] = True
        else:
            reward = Reward(
                value=-0.1,
                crash_penalty=-0.1,
                reason="Pipeline crashed. Check the logs for errors."
            )

        # Efficiency penalty for running pipeline too many times
        run_count = sum(
            1 for a in (self._episode_state.current_observation.actions_taken or [])
            if "run_pipeline" in a
        )
        if run_count > 5:
            reward.value -= 0.02
            reward.efficiency_penalty -= 0.02
            reward.reason += " Efficiency penalty: too many pipeline runs."

        obs = self._build_observation(logs)
        return obs, reward, False

    def _handle_submit(self, action: Action, info: dict):
        """submit → grade the final output and end the episode."""
        # Run pipeline one final time
        output, logs, success = self._pipeline.run()
        if success:
            self._last_output = output

        # Grade the output
        final_score, breakdown = graders.grade(
            pipeline=self._pipeline,
            actual_output=self._last_output,
            expected_output=self._task_def["expected_output"],
            expected_schema=self._task_def["expected_schema"],
            bugs=self._task_def["bugs"],
        )

        reward = Reward(
            value=final_score,
            schema_match=breakdown["schema_score"],
            data_match=breakdown["data_score"],
            progress=breakdown["bug_fix_score"],
            reason=(
                f"FINAL SCORE: {final_score:.2f} | "
                f"Schema: {breakdown['schema_score']:.2f} | "
                f"Data: {breakdown['data_score']:.2f} | "
                f"Bug fixes: {breakdown['bug_fix_score']:.2f} | "
                f"{breakdown['data_reason']}"
            )
        )

        info["success"] = final_score > 0.8
        info["message"] = f"Submitted. Final score: {final_score:.4f}"
        info["grader_breakdown"] = breakdown

        obs = self._build_observation(logs + [f"=== SUBMITTED. Score: {final_score:.4f} ==="])
        return obs, reward, True   # done = True

    # ──────────────────────────────────────────────────────────
    # Helper: Build an Observation from current pipeline state
    # ──────────────────────────────────────────────────────────
    def _build_observation(self, new_logs: list = None) -> Observation:
        stages = []
        for sid in self._pipeline.stage_order:
            s = self._pipeline.stages[sid]
            stages.append(StageInfo(
                stage_id=s.stage_id,
                stage_type=s.stage_type,
                description=s.description,
                code=s.code,
                inputs=s.inputs,
                status=s.status,
                error_message=s.error_message,
            ))

        current_output = []
        if self._last_output is not None:
            try:
                current_output = self._last_output.head(5).to_dict(orient="records")
            except Exception:
                pass

        expected_output = self._task_def["expected_output"].head(5).to_dict(orient="records")
        input_sample = self._pipeline.input_data.head(5).to_dict(orient="records")

        prev_actions = []
        if self._episode_state and self._episode_state.current_observation:
            prev_actions = list(self._episode_state.current_observation.actions_taken or [])

        steps_remaining = (
            self._task_def["max_steps"] - self._episode_state.step_count
            if self._episode_state else self._task_def["max_steps"]
        )

        return Observation(
            task_id=self._task_def["task_id"],
            task_description=self._task_def["description"],
            pipeline_stages=stages,
            input_data_sample=input_sample,
            current_output_sample=current_output,
            expected_output_sample=expected_output,
            expected_schema=self._task_def["expected_schema"],
            logs=list(new_logs or []),
            actions_taken=prev_actions,
            steps_remaining=steps_remaining,
            current_score=self._episode_state.total_reward if self._episode_state else 0.0,
        )