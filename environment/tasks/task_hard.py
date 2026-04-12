"""
task_hard.py — The Cascade (4 bugs, each hiding the next)

STORY:
A logistics company tracks shipments. Their pipeline:
1. Loads and casts shipment records
2. Standardizes column names
3. Joins with warehouse location data
4. Classifies shipments as domestic/international
5. Produces a summary report by region

FOUR BUGS (each causes the next):
Bug 1 (stage_2): Wrong column rename — 'warehouse_id' → 'wh_id'
                  breaks stage_3 join completely

Bug 2 (stage_3): Wrong join key — uses 'wh_id' but warehouses has 'warehouse_id'
                  even after fixing bug 1, join logic is still wrong

Bug 3 (stage_4): Wrong distance threshold — > 500 instead of > 1000
                  causes wrong domestic/international classification

Bug 4 (stage_4): Wrong column used for classification — uses 'distance_km'
                  but after join some rows have NULL distance due to bug 2
                  agent must also add a fillna() to handle NULLs

AGENT MUST:
1. Fix stage_2 column rename
2. Fix stage_3 join key
3. Fix stage_4 threshold AND handle NULLs
Must fix in order — later fixes are masked by earlier bugs.

DIFFICULTY: Hard
- 4 bugs across 5 stages
- Cascading failures
- NULL propagation adds extra complexity
- No single error message reveals all bugs
"""

import pandas as pd
from environment.pipeline import Pipeline, PipelineStage


def create_task() -> dict:

    # ── Input Data ──────────────────────────────────────────
    shipments = pd.DataFrame({
        "shipment_id":  [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
        "warehouse_id": ["WH01", "WH02", "WH01", "WH03", "WH02", "WH03", "WH01", "WH02"],
        "destination":  ["Mumbai", "London", "Delhi", "New York", "Chennai", "Dubai", "Pune", "Berlin"],
        "weight_kg":    [10.5, 25.0, 8.0, 50.0, 15.0, 30.0, 12.0, 22.0],
        "distance_km":  [200, 7000, 350, 12000, 180, 5000, 150, 6500],
        "priority":     ["standard", "express", "standard", "express",
                         "standard", "express", "standard", "express"],
    })

    input_data = shipments.copy()

    # ── Pipeline Stages ──────────────────────────────────────

    stage1 = PipelineStage(
        stage_id="stage_1",
        stage_type="transform",
        description="Load shipment data and cast numeric types",
        code="""
df['weight_kg'] = df['weight_kg'].astype(float)
df['distance_km'] = df['distance_km'].astype(float)
result = df
""",
        inputs=[]
    )

    # BUG 1: Wrong rename — wh_id breaks downstream join
    stage2 = PipelineStage(
        stage_id="stage_2",
        stage_type="transform",
        description="Standardize column names for downstream processing",
        code="""
df = df.rename(columns={'warehouse_id': 'wh_id'})  # BUG 1: should keep as 'warehouse_id'
result = df
""",
        inputs=["stage_1"]
    )

    # BUG 2: Wrong join key — left_on uses 'warehouse_id' but after bug1 it is 'wh_id'
    stage3 = PipelineStage(
        stage_id="stage_3",
        stage_type="transform",
        description="Join shipments with warehouse region and city data",
        code="""
warehouses = pd.DataFrame({
    'warehouse_id': ['WH01', 'WH02', 'WH03'],
    'region':       ['North India', 'South India', 'West India'],
    'city':         ['Delhi', 'Chennai', 'Mumbai'],
    'hub_type':     ['primary', 'primary', 'secondary'],
})
# BUG 2: left_on='warehouse_id' but column was renamed to 'wh_id' in stage_2
result = df.merge(warehouses, left_on='warehouse_id', right_on='warehouse_id', how='left')
""",
        inputs=["stage_2"]
    )

    # BUG 3: Wrong threshold (>500 instead of >1000)
    # BUG 4: No fillna for NULLs caused by broken join
    stage4 = PipelineStage(
        stage_id="stage_4",
        stage_type="transform",
        description="Classify shipments as domestic or international, then calculate revenue",
        code="""
# BUG 3: threshold should be > 1000, not > 500
# BUG 4: missing fillna(0) for distance_km NULLs from broken join
df['shipment_type'] = df['distance_km'].apply(
    lambda x: 'international' if x > 500 else 'domestic'
)
df['shipping_cost'] = df.apply(
    lambda row: row['weight_kg'] * (15 if row['shipment_type'] == 'international' else 5),
    axis=1
)
result = df
""",
        inputs=["stage_3"]
    )

    stage5 = PipelineStage(
        stage_id="stage_5",
        stage_type="aggregate",
        description="Summarize by region and shipment type: count, weight, cost",
        code="""
result = df.groupby(['region', 'shipment_type']).agg(
    shipment_count=('shipment_id', 'count'),
    total_weight=('weight_kg', 'sum'),
    total_cost=('shipping_cost', 'sum')
).reset_index().sort_values(['region', 'shipment_type']).reset_index(drop=True)
""",
        inputs=["stage_4"]
    )

    pipeline = Pipeline(
        stages=[stage1, stage2, stage3, stage4, stage5],
        input_data=input_data
    )

    # ── Expected Output ──────────────────────────────────────
    # After fixing ALL bugs:
    # Domestic (<=1000km): 1001(200,WH01,North), 1003(350,WH01,North),
    #                      1005(180,WH02,South), 1007(150,WH01,North)
    # International (>1000km): 1002(7000,WH02,South), 1004(12000,WH03,West),
    #                           1006(5000,WH03,West), 1008(6500,WH02,South)
    #
    # North India domestic: 1001(10.5), 1003(8.0), 1007(12.0) → count=3, weight=30.5, cost=152.5
    # South India domestic: 1005(15.0) → count=1, weight=15.0, cost=75.0
    # South India international: 1002(25.0), 1008(22.0) → count=2, weight=47.0, cost=705.0
    # West India international: 1004(50.0), 1006(30.0) → count=2, weight=80.0, cost=1200.0

    expected_output = pd.DataFrame({
        "region":         ["North India", "South India", "South India", "West India"],
        "shipment_type":  ["domestic",    "domestic",    "international", "international"],
        "shipment_count": [3,             1,             2,               2],
        "total_weight":   [30.5,          15.0,          47.0,            80.0],
        "total_cost":     [152.5,         75.0,          705.0,           1200.0],
    })

    expected_schema = {
        "region":         "object",
        "shipment_type":  "object",
        "shipment_count": "int64",
        "total_weight":   "float64",
        "total_cost":     "float64",
    }

    return {
        "task_id": "hard_cascade_bugs",
        "title": "Fix the Cascading Bugs in Logistics Shipment Pipeline",
        "description": (
            "A logistics company's shipment reporting pipeline is producing wrong output. "
            "There are multiple bugs: a column rename breaks a downstream join causing NULL regions, "
            "the join key is incorrect, the distance threshold for domestic vs international "
            "classification is wrong, and NULL values from the broken join need to be handled. "
            "Fix all bugs in the correct order — fixing later stages first won't help "
            "if earlier bugs still corrupt the data."
        ),
        "difficulty": "hard",
        "pipeline": pipeline,
        "expected_output": expected_output,
        "expected_schema": expected_schema,
       "bugs": [
    "stage_2_wrong_column_rename",
    "stage_3_wrong_join_key",
    "stage_4_wrong_threshold",
    "stage_4_null_handling",
],
        "hint": (
            "Run the pipeline first. Inspect stage outputs carefully. "
            "Think about what data flows from one stage to the next. "
            "Fix bugs from earliest stage to latest."
        ),
        "max_steps": 30,
    }