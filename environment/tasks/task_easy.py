"""
task_easy.py — The Type Mismatch Bug

STORY:
A retail shop tracks daily sales. Their pipeline processes
sales data and calculates total revenue per product.

THE BUG:
Stage 2 casts the 'price' column to int instead of float.
So ₹99.99 becomes ₹99, ₹149.50 becomes ₹149.
All revenue totals are slightly wrong.

WHAT AGENT MUST DO:
Find stage_2, change the type cast from int → float, verify fix.

DIFFICULTY: Easy
- There IS an error message (the totals don't match)
- Bug is in one place
- Fix is one line change
"""

import pandas as pd
from environment.pipeline import Pipeline, PipelineStage


def create_task() -> dict:
    """
    Returns the full task definition:
    - pipeline (with bug)
    - input data
    - expected output
    - task metadata
    """

    # ── Input Data ──────────────────────────────────────────
    # Raw sales records for a shop
    input_data = pd.DataFrame({
        "sale_id":    [1, 2, 3, 4, 5, 6],
        "product":    ["Shirt", "Pants", "Shirt", "Shoes", "Pants", "Shoes"],
        "price":      ["599.99", "1299.50", "599.99", "2499.00", "1299.50", "2499.00"],
        "quantity":   [2, 1, 3, 1, 2, 1],
        "date":       ["2024-01-01", "2024-01-01", "2024-01-02",
                       "2024-01-02", "2024-01-03", "2024-01-03"]
    })

    # ── Pipeline Stages (with bug) ───────────────────────────

    # Stage 1: Parse dates (correct)
    stage1 = PipelineStage(
        stage_id="stage_1",
        stage_type="transform",
        description="Parse date column to datetime format",
        code="""
df['date'] = pd.to_datetime(df['date'])
result = df
""",
        inputs=[]
    )

    # Stage 2: Cast types — BUG HERE! int instead of float for price
    stage2 = PipelineStage(
        stage_id="stage_2",
        stage_type="transform",
        description="Cast columns to correct data types",
        code="""
df['price'] = df['price'].astype(int)    # BUG: should be float, not int!
df['quantity'] = df['quantity'].astype(int)
result = df
""",
        inputs=["stage_1"]
    )

    # Stage 3: Calculate revenue (correct logic, but wrong because of bug above)
    stage3 = PipelineStage(
        stage_id="stage_3",
        stage_type="transform",
        description="Calculate revenue = price * quantity for each sale",
        code="""
df['revenue'] = df['price'] * df['quantity']
result = df
""",
        inputs=["stage_2"]
    )

    # Stage 4: Aggregate by product (correct)
    stage4 = PipelineStage(
        stage_id="stage_4",
        stage_type="aggregate",
        description="Sum total revenue per product",
        code="""
result = df.groupby('product')['revenue'].sum().reset_index()
result.columns = ['product', 'total_revenue']
result = result.sort_values('product').reset_index(drop=True)
""",
        inputs=["stage_3"]
    )

    pipeline = Pipeline(
        stages=[stage1, stage2, stage3, stage4],
        input_data=input_data
    )

    # ── Expected Output (what it SHOULD look like) ───────────
    # Calculated with correct float prices
    expected_output = pd.DataFrame({
        "product": ["Pants", "Shirt", "Shoes"],
        "total_revenue": [
            1299.50 * 1 + 1299.50 * 2,    # 3898.50
            599.99 * 2 + 599.99 * 3,       # 2999.95
            2499.00 * 1 + 2499.00 * 1,     # 4998.00
        ]
    })

    expected_schema = {
        "product": "object",
        "total_revenue": "float64"
    }

    return {
        "task_id": "easy_type_mismatch",
        "title": "Fix the Type Mismatch in Sales Pipeline",
        "description": (
            "A retail shop's sales pipeline calculates total revenue per product. "
            "The pipeline runs without crashing, but the revenue totals are wrong. "
            "Prices like ₹599.99 are being truncated to ₹599. "
            "Find and fix the stage that is casting prices to the wrong type."
        ),
        "difficulty": "easy",
        "pipeline": pipeline,
        "expected_output": expected_output,
        "expected_schema": expected_schema,
        "bugs": ["stage_2_wrong_type_cast"],
        "hint": "Check how data types are being cast in the pipeline stages.",
        "max_steps": 15,
    }