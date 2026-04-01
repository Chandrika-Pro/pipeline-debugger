"""
pipeline.py — The Juice Factory Machine.

This file actually RUNS the data pipeline.

Think of it like this:
- You give it a pipeline definition (list of stages with code)
- You give it input data (raw oranges)
- It runs each stage one by one (wash → squeeze → filter → bottle)
- It returns the output data + any errors that happened

The AI agent can MODIFY stages (fix bugs), then RE-RUN the pipeline
to see if its fix worked.
"""

import pandas as pd
import traceback
from typing import Dict, List, Any, Tuple, Optional
from copy import deepcopy


class PipelineStage:
    """
    One stage in the pipeline — one machine in the factory.

    Each stage has:
    - an id (e.g. "stage_1")
    - a type (e.g. "transform", "filter", "aggregate")
    - Python code that transforms a DataFrame
    """

    def __init__(
        self,
        stage_id: str,
        stage_type: str,
        description: str,
        code: str,
        inputs: List[str] = None
    ):
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.description = description
        self.code = code                        # Python code as a string
        self.inputs = inputs or []              # which stages feed into this
        self.status = "unknown"
        self.error_message = None

    def run(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Run this stage.

        dataframes = all DataFrames produced so far (by stage_id)
        Returns the new DataFrame produced by this stage.

        The code string is executed with 'df' as input.
        Example code:
            df['price'] = df['price'].astype(float)
            result = df
        """
        # Collect input DataFrames
        if not self.inputs:
            # No inputs = this is the first stage, use 'input' key
            df = dataframes.get("input", pd.DataFrame()).copy()
        elif len(self.inputs) == 1:
            df = dataframes[self.inputs[0]].copy()
        else:
            # Multiple inputs = merge them
            df = dataframes[self.inputs[0]].copy()
            for inp in self.inputs[1:]:
                df = df.merge(dataframes[inp], how="left")

        # Execute the stage code
        local_vars = {"df": df, "pd": pd}
        exec(self.code, {}, local_vars)

        # The code must assign to 'result'
        if "result" not in local_vars:
            raise ValueError(
                f"Stage '{self.stage_id}' code must assign output to 'result'. "
                f"Example: result = df"
            )

        self.status = "ok"
        self.error_message = None
        return local_vars["result"]

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "stage_type": self.stage_type,
            "description": self.description,
            "code": self.code,
            "inputs": self.inputs,
            "status": self.status,
            "error_message": self.error_message,
        }


class Pipeline:
    """
    The full data pipeline — the entire juice factory.

    Contains multiple PipelineStages run in order.
    Tracks which stages passed/failed.
    Agent can modify stages and re-run.
    """

    def __init__(self, stages: List[PipelineStage], input_data: pd.DataFrame):
        self.stages = {s.stage_id: s for s in stages}
        self.stage_order = [s.stage_id for s in stages]
        self.input_data = input_data.copy()
        self.last_run_results: Dict[str, pd.DataFrame] = {}
        self.logs: List[str] = []

    def run(self) -> Tuple[Optional[pd.DataFrame], List[str], bool]:
        """
        Run the entire pipeline from start to finish.

        Returns:
        - output DataFrame (or None if pipeline crashed)
        - list of log messages
        - success (True/False)
        """
        self.logs = []
        self.last_run_results = {"input": self.input_data.copy()}
        success = True
        output = None

        for stage_id in self.stage_order:
            stage = self.stages[stage_id]
            self.logs.append(f"▶ Running {stage_id} ({stage.stage_type})...")

            try:
                result = stage.run(self.last_run_results)
                self.last_run_results[stage_id] = result
                stage.status = "ok"
                self.logs.append(f"  ✓ {stage_id} completed. Output shape: {result.shape}")
                output = result

            except Exception as e:
                stage.status = "error"
                stage.error_message = str(e)
                self.logs.append(f"  ✗ {stage_id} FAILED: {e}")
                self.logs.append(f"    Traceback: {traceback.format_exc().splitlines()[-2]}")
                success = False
                break  # Stop pipeline on first failure

        return output, self.logs, success

    def fix_stage_code(self, stage_id: str, new_code: str) -> bool:
        """Replace a stage's code with new code. Returns True if stage exists."""
        if stage_id not in self.stages:
            return False
        self.stages[stage_id].code = new_code
        self.stages[stage_id].status = "unknown"
        self.stages[stage_id].error_message = None
        return True

    def fix_schema(self, stage_id: str, column: str, new_type: str) -> bool:
        """
        Add a type-cast line to a stage's code.
        This is a convenience action so the agent doesn't have to write full code.
        """
        if stage_id not in self.stages:
            return False

        type_map = {
            "float": "float",
            "int": "int",
            "str": "str",
            "string": "str",
            "datetime": "pd.to_datetime",
        }

        if new_type not in type_map:
            return False

        if new_type == "datetime":
            cast_line = f"df['{column}'] = pd.to_datetime(df['{column}'])"
        else:
            cast_line = f"df['{column}'] = df['{column}'].astype({type_map[new_type]})"

        # Prepend the cast to existing code
        existing = self.stages[stage_id].code
        self.stages[stage_id].code = cast_line + "\n" + existing
        self.stages[stage_id].status = "unknown"
        return True

    def reorder_stages(self, new_order: List[str]) -> bool:
        """Change the order stages run in."""
        if set(new_order) != set(self.stage_order):
            return False
        self.stage_order = new_order
        return True

    def get_stage_info(self, stage_id: str) -> Optional[dict]:
        """Get full details of a specific stage (for inspect_stage action)."""
        if stage_id not in self.stages:
            return None
        return self.stages[stage_id].to_dict()

    def get_output_sample(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get first n rows of the latest output as a list of dicts."""
        if not self.last_run_results:
            return []
        # Get last stage's output
        last_stage_id = self.stage_order[-1]
        if last_stage_id not in self.last_run_results:
            # Pipeline failed mid-way, return last successful output
            for sid in reversed(self.stage_order):
                if sid in self.last_run_results:
                    df = self.last_run_results[sid]
                    return df.head(n).to_dict(orient="records")
            return []
        df = self.last_run_results[last_stage_id]
        return df.head(n).to_dict(orient="records")

    def clone(self) -> "Pipeline":
        """Deep copy of the pipeline (used for reset)."""
        stages = [
            PipelineStage(
                stage_id=s.stage_id,
                stage_type=s.stage_type,
                description=s.description,
                code=s.code,
                inputs=list(s.inputs),
            )
            for s in self.stages.values()
        ]
        return Pipeline(stages, self.input_data.copy())