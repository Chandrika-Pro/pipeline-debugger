"""
task_hard.py — The Cascade (3 bugs, each hiding the next)

STORY:
A logistics company tracks shipments. Their pipeline:
1. Loads shipment records
2. Joins with warehouse location data
3. Classifies shipments as domestic/international
4. Produces a summary report by region

THREE BUGS (each causes the next):
Bug 1 (stage_2): Column rename uses wrong name → 'warehouse_id' renamed to 'wh_id'
                  but stage_3 join expects 'warehouse_id'
                  → Stage 3 join produces all NULLs

Bug 2 (stage_3): Even if join worked, the join key is wrong ('id' vs 'warehouse_id')
                  → NULLs in region column

Bug 3 (stage_4): Classifier uses wrong threshold — classifies as 'international'
                  if distance > 500, but should be > 1000
                  → Wrong domestic/international split in final report

AGENT MUST:
1. Realize stage_3 fails because stage_2 renamed a column badly → fix stage_2
2. Then realize stage_3 has wrong join key → fix stage_3
3. Then realize stage_4 threshold is wrong → fix stage_4
Must fix in order — fixing stage_4 first won't help if stage_3 still produces NULLs.

DIFFICULTY: Hard
- 3 bugs across 4 stages
- Each bug is partially masked by the next
- No single error message tells the full story
- Requires reasoning about data flow
"""

import pandas as pd
from environment.pipeline import Pipeline, PipelineStage


def create_task() -> dict:

    # ── Input Data ──────────────────────────────────────────

    # Shipment records
    shipments = pd.DataFrame({
        "shipment_id":  [1001, 1002, 1003, 1004, 1005, 1006],
        "warehouse_id": ["WH01", "WH02", "WH01", "WH03", "WH02", "WH03"],
        "destination":  ["Mumbai", "London", "Delhi", "New York", "Chennai", "Dubai"],
        "weight_kg":    [10.5, 25.0, 8.0, 50.0, 15.0, 30.0],
        "distance_km":  [200, 7000, 350, 12000, 180, 5000],
    })

    # Warehouse locations (to be joined)
    warehouses = pd.DataFrame({
        "warehouse_id": ["WH01", "WH02", "WH03"],
        "region":       ["North India", "South India", "West India"],
        "city":         ["Delhi", "Chennai", "Mumbai"],
    })

    # We store warehouses as a second input — merged at pipeline start
    input_data = shipments.copy()
    # We'll pass warehouses via pipeline metadata

    # ── Pipeline Stages ──────────────────────────────────────

    stage1 = PipelineStage(
        stage_id="stage_1",
        stage_type="transform",
        description="Load shipment data and cast types",
        code="""
df['weight_kg'] = df['weight_kg'].astype(float)
df['distance_km'] = df['distance_km'].astype(float)
result = df
""",
        inputs=[]
    )

    # BUG 1: Renames warehouse_id to wh_id — breaks stage_3 join
    stage2 = PipelineStage(
        stage_id="stage_2",
        stage_type="transform",
        description="Standardize column names for downstream processing",
        code="""
df = df.rename(columns={'warehouse_id': 'wh_id'})   # BUG: should keep as 'warehouse_id'
result = df
""",
        inputs=["stage_1"]
    )

    # BUG 2: Join key is wrong — uses 'id' which doesn't exist
    # Even without Bug 1, this join key is wrong → all NULLs in region
    stage3 = PipelineStage(
        stage_id="stage_3",
        stage_type="transform",
        description="Join shipments with warehouse region data",
        code="""
warehouses = pd.DataFrame({
    'warehouse_id': ['WH01', 'WH02', 'WH03'],
    'region':       ['North India', 'South India', 'West India'],
    'city':         ['Delhi', 'Chennai', 'Mumbai'],
})
# BUG: left_on should be 'warehouse_id' (or 'wh_id' after bug1), right_on correct
result = df.merge(warehouses, left_on='warehouse_id', right_on='warehouse_id', how='left')
""",
        inputs=["stage_2"]
    )

    # BUG 3: Threshold wrong — > 500 instead of > 1000
    stage4 = PipelineStage(
        stage_id="stage_4",
        stage_type="transform",
        description="Classify shipments as domestic or international based on distance",
        code="""
df['shipment_type'] = df['distance_km'].apply(
    lambda x: 'international' if x > 500 else 'domestic'   # BUG: should be > 1000
)
result = df
""",
        inputs=["stage_3"]
    )

    stage5 = PipelineStage(
        stage_id="stage_5",
        stage_type="aggregate",
        description="Summarize shipment counts and weight by region and type",
        code="""
result = df.groupby(['region', 'shipment_type']).agg(
    shipment_count=('shipment_id', 'count'),
    total_weight=('weight_kg', 'sum')
).reset_index().sort_values(['region', 'shipment_type']).reset_index(drop=True)
""",
        inputs=["stage_4"]
    )

    pipeline = Pipeline(
        stages=[stage1, stage2, stage3, stage4, stage5],
        input_data=input_data
    )

    # ── Expected Output ──────────────────────────────────────
    # After fixing all 3 bugs:
    # Domestic (<=1000km): 1001(200), 1003(350), 1005(180) → WH01,WH01,WH02
    # International (>1000km): 1002(7000), 1004(12000), 1006(5000) → WH02,WH03,WH03
    expected_output = pd.DataFrame({
        "region":         ["North India", "South India", "South India", "West India"],
        "shipment_type":  ["domestic",    "domestic",    "international", "international"],
        "shipment_count": [2,             1,             1,               2],
        "total_weight":   [18.5,          15.0,          25.0,            80.0],
    })

    expected_schema = {
        "region": "object",
        "shipment_type": "object",
        "shipment_count": "int64",
        "total_weight": "float64",
    }

    return {
        "task_id": "hard_cascade_bugs",
        "title": "Fix the Cascading Bugs in Logistics Shipment Pipeline",
        "description": (
            "A logistics company's shipment reporting pipeline is producing wrong output. "
            "There are multiple bugs: a column rename is breaking a downstream join, "
            "the join key is incorrect causing NULL regions, and the distance threshold "
            "for classifying domestic vs international shipments is wrong. "
            "You must find and fix all bugs. Note: fixing later stages won't help "
            "if earlier bugs are still present — order matters."
        ),
        "difficulty": "hard",
        "pipeline": pipeline,
        "expected_output": expected_output,
        "expected_schema": expected_schema,
        "bugs": [
            "stage_2_wrong_column_rename",
            "stage_3_wrong_join_key",
            "stage_4_wrong_threshold",
        ],
        "hint": "Run the pipeline first. Then inspect each stage's output carefully. Think about what data flows from one stage to the next.",
        "max_steps": 30,
    }