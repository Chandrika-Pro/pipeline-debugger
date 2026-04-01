"""
models.py — The "Forms" of our environment.
Using Python dataclasses (built-in).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class StageInfo:
    stage_id: str
    stage_type: str
    description: str
    code: Optional[str] = None
    inputs: List[str] = field(default_factory=list)
    status: str = "unknown"
    error_message: Optional[str] = None


@dataclass
class Observation:
    task_id: str
    task_description: str
    pipeline_stages: List[StageInfo]
    input_data_sample: List[Dict[str, Any]]
    current_output_sample: List[Dict[str, Any]]
    expected_output_sample: List[Dict[str, Any]]
    expected_schema: Dict[str, str]
    logs: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    steps_remaining: int = 20
    current_score: float = 0.0


@dataclass
class Action:
    action_type: str
    stage_id: Optional[str] = None
    new_code: Optional[str] = None
    column_name: Optional[str] = None
    new_type: Optional[str] = None
    new_condition: Optional[str] = None
    new_order: Optional[List[str]] = None
    reasoning: Optional[str] = None

    VALID_TYPES = frozenset({
        "inspect_stage", "fix_transform", "fix_schema",
        "fix_filter", "reorder_stages", "run_pipeline", "submit"
    })

    def __post_init__(self):
        if self.action_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid action_type '{self.action_type}'")


@dataclass
class Reward:
    value: float = 0.0
    schema_match: float = 0.0
    data_match: float = 0.0
    progress: float = 0.0
    efficiency_penalty: float = 0.0
    crash_penalty: float = 0.0
    reason: str = ""

    def __post_init__(self):
        self.value = max(-1.0, min(1.0, self.value))


@dataclass
class EpisodeState:
    task_id: str
    step_count: int = 0
    done: bool = False
    total_reward: float = 0.0
    current_observation: Optional[Observation] = None
    pipeline_fixed: bool = False
    bugs_fixed: List[str] = field(default_factory=list)
    bugs_remaining: List[str] = field(default_factory=list)