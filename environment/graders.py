"""
graders.py — The Answer Key

This file compares what the agent produced vs what was expected,
and gives a score between 0.0 and 1.0.

Think of a teacher grading an exam:
- Is the schema correct? (column names, types)
- Does the data match? (values close enough?)
- Were bugs actually fixed?

Scoring breakdown:
  0.3 → schema matches (correct columns and types)
  0.5 → data matches (values correct within tolerance)
  0.2 → all bugs fixed (confirmed by checking pipeline stage code)

Total: 1.0 maximum
"""

import pandas as pd
import numpy as np
from typing import Optional


def score_schema(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    expected_schema: dict
) -> tuple[float, str]:
    """
    Score how well the output schema matches the expected schema.
    Returns (score 0.0-0.3, reason string)
    """
    if actual is None or actual.empty:
        return 0.0, "No output produced."

    score = 0.0
    reasons = []

    # Check column names
    actual_cols = set(actual.columns)
    expected_cols = set(expected.columns)

    if actual_cols == expected_cols:
        score += 0.15
        reasons.append("✓ Column names match.")
    else:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        if missing:
            reasons.append(f"✗ Missing columns: {missing}")
        if extra:
            reasons.append(f"✗ Extra columns: {extra}")
        # Partial credit if at least some columns match
        overlap = len(actual_cols & expected_cols) / len(expected_cols)
        score += 0.15 * overlap

    # Check data types
    type_matches = 0
    for col, expected_type in expected_schema.items():
        if col in actual.columns:
            actual_type = str(actual[col].dtype)
            if actual_type == expected_type or (
                "float" in actual_type and "float" in expected_type
            ) or (
                "int" in actual_type and "int" in expected_type
            ):
                type_matches += 1

    type_score = type_matches / len(expected_schema) if expected_schema else 1.0
    score += 0.15 * type_score
    reasons.append(f"✓ {type_matches}/{len(expected_schema)} column types correct.")

    return round(score, 4), " ".join(reasons)


def score_data(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    tolerance: float = 0.01
) -> tuple[float, str]:
    """
    Score how closely the data values match expected values.
    Returns (score 0.0-0.5, reason string)
    """
    if actual is None or actual.empty:
        return 0.0, "No output data to compare."

    if set(actual.columns) != set(expected.columns):
        return 0.0, "Cannot compare data — column names don't match."

    # Reorder columns to match
    try:
        actual = actual[expected.columns].reset_index(drop=True)
        expected = expected.reset_index(drop=True)
    except Exception:
        return 0.0, "Could not align columns for comparison."

    # Check row count
    if len(actual) != len(expected):
        row_ratio = min(len(actual), len(expected)) / max(len(actual), len(expected))
        return round(0.1 * row_ratio, 4), (
            f"✗ Row count mismatch: got {len(actual)}, expected {len(expected)}."
        )

    # Sort both DataFrames the same way for fair comparison
    try:
        sort_cols = list(expected.select_dtypes(include="object").columns)
        if sort_cols:
            actual = actual.sort_values(sort_cols).reset_index(drop=True)
            expected = expected.sort_values(sort_cols).reset_index(drop=True)
    except Exception:
        pass

    # Compare each column
    total_cells = 0
    correct_cells = 0

    for col in expected.columns:
        if expected[col].dtype in [object, "string"]:
            matches = (actual[col].astype(str) == expected[col].astype(str)).sum()
        else:
            try:
                diff = (actual[col].astype(float) - expected[col].astype(float)).abs()
                matches = (diff <= tolerance * expected[col].astype(float).abs().clip(lower=1)).sum()
            except Exception:
                matches = 0

        correct_cells += matches
        total_cells += len(expected)

    accuracy = correct_cells / total_cells if total_cells > 0 else 0.0
    score = round(0.5 * accuracy, 4)
    reason = f"{'✓' if accuracy > 0.95 else '~'} Data accuracy: {accuracy:.1%} ({correct_cells}/{total_cells} cells correct)."

    return score, reason


def score_bugs_fixed(
    pipeline,
    bugs: list[str]
) -> tuple[float, str]:
    """
    Check if specific bugs have been fixed by inspecting pipeline stage code.
    Returns (score 0.0-0.2, reason string)
    """
    if not bugs:
        return 0.2, "No bugs to check."

    fixed_count = 0
    reasons = []

    for bug in bugs:

        # ── EASY TASK ──────────────────────────────────────
        if bug == "stage_2_wrong_type_cast":
            # stage_2 should cast price as float not int
            code = pipeline.stages.get("stage_2", None)
            if code:
                price_line = ""
                for line in code.code.split("\n"):
                    if "price" in line and "astype" in line:
                        price_line = line
                        break
                if "astype(float)" in price_line:
                    fixed_count += 1
                    reasons.append("✓ Type cast bug fixed.")
                else:
                    reasons.append("✗ Type cast bug still present.")

        # ── MEDIUM TASK ────────────────────────────────────
        elif bug == "stage_3_wrong_date_boundary":
            # stage_3 should use 2024-02-01 not 2024-01-31
            code = pipeline.stages.get("stage_3", None)
            if code:
                if "2024-01-31" not in code.code:
                    fixed_count += 1
                    reasons.append("✓ Date boundary bug fixed.")
                else:
                    reasons.append("✗ Date boundary still uses 2024-01-31.")

        # ── HARD TASK ──────────────────────────────────────
        elif bug == "stage_2_wrong_column_rename":
            # stage_2 should NOT rename warehouse_id to wh_id
            code = pipeline.stages.get("stage_2", None)
            if code:
                if "wh_id" not in code.code:
                    fixed_count += 1
                    reasons.append("✓ Column rename bug fixed.")
                else:
                    reasons.append("✗ Column rename bug still present (wh_id).")

        elif bug == "stage_3_wrong_join_key":
            # stage_3 join should use warehouse_id correctly
            code = pipeline.stages.get("stage_3", None)
            if code:
                if "warehouse_id" in code.code and "wh_id" not in code.code:
                    fixed_count += 1
                    reasons.append("✓ Join key bug fixed.")
                else:
                    reasons.append("✗ Join key bug still present.")

        elif bug == "stage_4_wrong_threshold":
            # stage_4 threshold should be > 1000 not > 500
            code = pipeline.stages.get("stage_4", None)
            if code:
                if "> 1000" in code.code and "> 500" not in code.code:
                    fixed_count += 1
                    reasons.append("✓ Threshold bug fixed.")
                else:
                    reasons.append("✗ Threshold bug still uses > 500.")

        elif bug == "stage_4_null_handling":
            # stage_4 should handle NULLs with fillna
            code = pipeline.stages.get("stage_4", None)
            if code:
                if "fillna" in code.code or "dropna" in code.code or "notnull" in code.code:
                    fixed_count += 1
                    reasons.append("✓ NULL handling added.")
                else:
                    reasons.append("✗ NULL handling still missing.")

    score = round(0.2 * (fixed_count / len(bugs)), 4)
    return score, " ".join(reasons)


def grade(
    pipeline,
    actual_output: Optional[pd.DataFrame],
    expected_output: pd.DataFrame,
    expected_schema: dict,
    bugs: list[str]
) -> tuple[float, dict]:
    """
    Full grader — combines all three scoring components.

    Returns:
    - final_score (0.0 to 1.0)
    - breakdown dict with details
    """
    schema_score, schema_reason = score_schema(actual_output, expected_output, expected_schema)
    data_score, data_reason     = score_data(actual_output, expected_output)
    bug_score, bug_reason       = score_bugs_fixed(pipeline, bugs)

    final_score = round(schema_score + data_score + bug_score, 4)
    final_score = min(final_score, 1.0)

    return final_score, {
        "final_score":   final_score,
        "schema_score":  schema_score,
        "data_score":    data_score,
        "bug_fix_score": bug_score,
        "schema_reason": schema_reason,
        "data_reason":   data_reason,
        "bug_reason":    bug_reason,
    }