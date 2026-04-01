"""
task_medium.py — The Silent Logic Bug

STORY:
An e-commerce company tracks orders. Their pipeline
filters orders from "last month" for a monthly report.

THE BUG:
The date filter uses >= on the start date instead of >.
So orders from exactly the 1st of last month are included
when they belong to the month before.

WHAT MAKES THIS HARDER:
- Pipeline runs with NO errors
- No crash, no exception
- The output LOOKS normal
- Agent must compare output vs expected carefully
- Then find the logic bug, not a type error

DIFFICULTY: Medium
- No error messages to guide the agent
- Agent must reason about data values
- Fix is small but requires understanding date logic
"""

import pandas as pd
from environment.pipeline import Pipeline, PipelineStage


def create_task() -> dict:

    # ── Input Data ──────────────────────────────────────────
    # Orders across two months
    input_data = pd.DataFrame({
        "order_id":  [101, 102, 103, 104, 105, 106, 107, 108],
        "customer":  ["Alice", "Bob", "Alice", "Charlie", "Bob",
                      "Diana", "Alice", "Charlie"],
        "amount":    [250.0, 180.0, 320.0, 90.0, 410.0, 150.0, 280.0, 500.0],
        "order_date": [
            "2024-01-31",   # LAST day of Jan → should NOT be in Feb report
            "2024-02-01",   # First day of Feb → BUG includes this incorrectly
            "2024-02-05",
            "2024-02-10",
            "2024-02-15",
            "2024-02-20",
            "2024-02-25",
            "2024-02-28",
        ],
        "status": ["completed", "completed", "completed", "pending",
                   "completed", "completed", "completed", "completed"]
    })

    # ── Pipeline Stages ──────────────────────────────────────

    stage1 = PipelineStage(
        stage_id="stage_1",
        stage_type="transform",
        description="Parse order_date to datetime",
        code="""
df['order_date'] = pd.to_datetime(df['order_date'])
result = df
""",
        inputs=[]
    )

    stage2 = PipelineStage(
        stage_id="stage_2",
        stage_type="filter",
        description="Filter to completed orders only",
        code="""
result = df[df['status'] == 'completed'].copy()
""",
        inputs=["stage_1"]
    )

    # BUG: >= should be > for start date, so that Jan 31 is excluded
    stage3 = PipelineStage(
        stage_id="stage_3",
        stage_type="filter",
        description="Filter orders from February 2024 (the current report month)",
        code="""
start = pd.Timestamp('2024-02-01')
end   = pd.Timestamp('2024-02-29')
result = df[(df['order_date'] >= start) & (df['order_date'] <= end)].copy()
# BUG: >= start should be > start to exclude Feb 1 (which is actually Jan's boundary)
# Wait — actually the real bug is that Jan 31 crept in.
# The correct filter should ALSO exclude the jan 31 record that has been mislabeled.
# Let's keep it simpler: the correct filter for Feb is month==2
""",
        inputs=["stage_2"]
    )

    # Correct stage3 re-defined with actual bug (wrong comparison for month boundary)
    stage3 = PipelineStage(
        stage_id="stage_3",
        stage_type="filter",
        description="Filter orders from February 2024 (the current report month)",
        code="""
start = pd.Timestamp('2024-01-31')   # BUG: should be 2024-02-01
end   = pd.Timestamp('2024-02-29')
result = df[(df['order_date'] >= start) & (df['order_date'] <= end)].copy()
""",
        inputs=["stage_2"]
    )

    stage4 = PipelineStage(
        stage_id="stage_4",
        stage_type="aggregate",
        description="Calculate total and average order value for the month",
        code="""
result = pd.DataFrame({
    'metric': ['total_orders', 'total_revenue', 'avg_order_value'],
    'value': [
        len(df),
        round(df['amount'].sum(), 2),
        round(df['amount'].mean(), 2)
    ]
})
""",
        inputs=["stage_3"]
    )

    pipeline = Pipeline(
        stages=[stage1, stage2, stage3, stage4],
        input_data=input_data
    )

    # ── Expected Output ──────────────────────────────────────
    # Only Feb orders (not Jan 31): orders 102-108 completed = 102,103,105,106,107,108
    # 102: 180, 103: 320, 105: 410, 106: 150, 107: 280, 108: 500
    # total = 1840, count = 6, avg = 306.67
    expected_output = pd.DataFrame({
        "metric": ["total_orders", "total_revenue", "avg_order_value"],
        "value":  [6.0, 1840.0, 306.67]
    })

    expected_schema = {
        "metric": "object",
        "value":  "float64"
    }

    return {
        "task_id": "medium_silent_filter_bug",
        "title": "Fix the Silent Filter Bug in Monthly Report Pipeline",
        "description": (
            "An e-commerce company generates a monthly orders report for February 2024. "
            "The pipeline runs without any errors, but the output numbers don't match "
            "the finance team's manual count. The report is showing one extra order "
            "from January. Find the date filter that is including wrong records "
            "and fix the boundary condition."
        ),
        "difficulty": "medium",
        "pipeline": pipeline,
        "expected_output": expected_output,
        "expected_schema": expected_schema,
        "bugs": ["stage_3_wrong_date_boundary"],
        "hint": "The pipeline runs cleanly. Compare the output carefully with expected. Focus on the date filter.",
        "max_steps": 20,
    }