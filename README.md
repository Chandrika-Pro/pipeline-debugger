---
title: Pipeline Debugger OpenEnv
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 🔧 Pipeline Debugger — OpenEnv Environment

An OpenEnv-compliant environment where AI agents debug **broken real-world data pipelines**.

Built for the Scaler School of Technology Hackathon — Round 1.

---

## 🧠 What Is This?

Every company runs **data pipelines** — automated sequences that clean, transform,
and aggregate raw data into useful reports. When these pipelines break, engineers
spend hours tracking down bugs. This environment simulates exactly that problem.

An AI agent is given a **broken pipeline** and must:
1. **Inspect** stages to understand what's going wrong
2. **Fix** the broken code
3. **Verify** by running the pipeline
4. **Submit** when confident the output matches the expected result

The environment provides **partial rewards** throughout the episode — not just a
binary pass/fail at the end — making it suitable for reinforcement learning.

---

## 🗂 Project Structure

```
openenv-pipeline-debugger/
├── environment/
│   ├── env.py          # Main environment class (reset/step/state)
│   ├── models.py       # Typed dataclass models (Observation, Action, Reward)
│   ├── pipeline.py     # Pipeline execution engine
│   ├── graders.py      # Task graders (scores 0.0–1.0)
│   └── tasks/
│       ├── task_easy.py    # Type mismatch bug
│       ├── task_medium.py  # Silent filter logic bug
│       └── task_hard.py    # 3 cascading bugs
├── baseline/
│   └── run_baseline.py # OpenAI baseline agent script
├── tests/
│   └── test_env.py     # 13 unit tests
├── app.py              # FastAPI server for HF Spaces
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔁 OpenEnv Interface

```python
from environment.env import PipelineDebuggerEnv
from environment.models import Action

env = PipelineDebuggerEnv(task_name="easy")  # or "medium" / "hard"

# Start a new episode
obs = env.reset()

# Take an action
obs, reward, done, info = env.step(Action(
    action_type="fix_transform",
    stage_id="stage_2",
    new_code="df['price'] = df['price'].astype(float)\nresult = df"
))

# Submit for final grading
obs, reward, done, info = env.step(Action(action_type="submit"))

# Get current state snapshot
state = env.state()
```

---

## 👁 Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Task identifier |
| `task_description` | str | Plain-English problem description |
| `pipeline_stages` | List[StageInfo] | All stages with code, status, errors |
| `input_data_sample` | List[dict] | First 5 rows of input data |
| `current_output_sample` | List[dict] | First 5 rows of current (broken) output |
| `expected_output_sample` | List[dict] | First 5 rows of expected correct output |
| `expected_schema` | Dict[str, str] | Column names → expected data types |
| `logs` | List[str] | Pipeline run logs and error messages |
| `actions_taken` | List[str] | History of agent actions this episode |
| `steps_remaining` | int | How many steps the agent has left |
| `current_score` | float | Cumulative reward so far |

Each `StageInfo` contains:
- `stage_id`, `stage_type`, `description`
- `code` — the Python transformation code (agent can read and fix this)
- `status` — `"ok"`, `"error"`, or `"unknown"`
- `error_message` — exception message if stage failed

---

## 🕹 Action Space

| `action_type` | Required Fields | Description |
|---|---|---|
| `inspect_stage` | `stage_id` | Reveal full details of a pipeline stage |
| `fix_transform` | `stage_id`, `new_code` | Replace a stage's transformation code |
| `fix_schema` | `stage_id`, `column_name`, `new_type` | Add a type cast to a stage |
| `fix_filter` | `stage_id`, `new_condition` | Replace a filter condition |
| `reorder_stages` | `new_order` | Change the order stages execute |
| `run_pipeline` | — | Execute pipeline and observe output |
| `submit` | — | Finalize fix and trigger grading |

All actions accept an optional `reasoning` string for chain-of-thought logging.

---

## 💰 Reward Function

Rewards are given at **every step** (not just at the end):

| Event | Reward |
|---|---|
| `fix_transform` / `fix_schema` applied | +0.02 |
| Pipeline runs clean after a fix | +0.05 |
| `run_pipeline` succeeds | +0.01 |
| `run_pipeline` crashes | **-0.10** |
| `inspect_stage` on nonexistent stage | -0.02 |
| Too many `run_pipeline` calls (>5) | -0.02 per extra |
| **submit** — schema correct | up to +0.30 |
| **submit** — data values correct | up to +0.50 |
| **submit** — bugs confirmed fixed | up to +0.20 |

**Maximum score per episode: 1.0**

---

## 📋 Tasks

### Task 1 — Easy: Type Mismatch (`easy_type_mismatch`)

**Domain:** Retail sales pipeline

**Story:** A shop's pipeline calculates total revenue per product. Prices like
₹599.99 are being truncated to ₹599 because one stage casts the price column
to `int` instead of `float`.

**Bug:** `stage_2` — `df['price'].astype(int)` should be `astype(float)`

**Max steps:** 15
**Expected GPT-4o score:** ~0.85–0.95
**Difficulty:** ⭐

---

### Task 2 — Medium: Silent Filter Bug (`medium_silent_filter_bug`)

**Domain:** E-commerce monthly reporting

**Story:** A monthly orders report shows wrong totals. The pipeline runs with
NO errors — no crashes, no exceptions. But the output includes one extra order
from the wrong month because the date boundary filter uses the wrong start date.

**Bug:** `stage_3` — filter start is `2024-01-31` should be `2024-02-01`

**Max steps:** 20
**Expected GPT-4o score:** ~0.55–0.70
**Difficulty:** ⭐⭐⭐

---

### Task 3 — Hard: Cascading Bugs (`hard_cascade_bugs`)

**Domain:** Logistics shipment reporting

**Story:** A logistics pipeline has THREE bugs, each masking the next:
1. `stage_2` renames `warehouse_id` → `wh_id`, breaking the downstream join
2. `stage_3` join uses wrong key, producing NULL regions
3. `stage_4` classifies shipments with threshold `> 500 km` instead of `> 1000 km`

Fixing bug 3 first does nothing if bugs 1 and 2 still corrupt the data.
The agent must reason about data flow and fix in the correct order.

**Max steps:** 30
**Expected GPT-4o score:** ~0.25–0.40
**Difficulty:** ⭐⭐⭐⭐⭐

---

## 📊 Baseline Scores

Tested with `gpt-4o` (temperature=0.1):

| Task | Score | Steps Used |
|---|---|---|
| easy | 0.9250 | 4 |
| medium | 0.6100 | 8 |
| hard | 0.3200 | 18 |
| **Average** | **0.6183** | — |

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone the repo
git clone https://huggingface.co/spaces/your-username/pipeline-debugger
cd pipeline-debugger

# Install dependencies
pip install -r requirements.txt

# Run the tests
python tests/test_env.py

# Start the API server
python app.py
# Server runs at http://localhost:7860
```

### Using the Environment Directly (Python)

```python
from environment.env import PipelineDebuggerEnv
from environment.models import Action

env = PipelineDebuggerEnv("easy")
obs = env.reset()

print(obs.task_description)
print(obs.current_output_sample)   # broken output
print(obs.expected_output_sample)  # what it should be

# Fix the bug
obs, reward, done, info = env.step(Action(
    action_type="fix_transform",
    stage_id="stage_2",
    new_code="df['price'] = df['price'].astype(float)\nresult = df",
    reasoning="Price should be float not int"
))

# Submit
obs, reward, done, info = env.step(Action("submit"))
print(f"Final score: {reward.value:.4f}")
```

### Using the HTTP API

```bash
# Start server
python app.py

# Reset (start new episode)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task": "easy",
    "action_type": "fix_schema",
    "stage_id": "stage_2",
    "column_name": "price",
    "new_type": "float"
  }'

# Get state
curl http://localhost:7860/state?task=easy
```

### Running the Baseline Script

```bash
export OPENAI_API_KEY=your_key_here

# Run all 3 tasks
python baseline/run_baseline.py

# Run a specific task with a specific model
python baseline/run_baseline.py --task easy --model gpt-4o-mini
```

### Docker

```bash
# Build
docker build -t pipeline-debugger .

# Run
docker run -p 7860:7860 pipeline-debugger

# With OpenAI key for baseline
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key pipeline-debugger
```

---

## 🧪 Running Tests

```bash
python tests/test_env.py
```

Expected output:
```
[TEST 1]  reset() returns valid Observation... ✓ PASSED
[TEST 2]  state() returns EpisodeState after reset... ✓ PASSED
...
Results: 13 passed, 0 failed
```

---

## 🏷 OpenEnv Validation

```bash
openenv validate
```

The `openenv.yaml` defines all metadata, observation/action spaces,
reward range, and task descriptions per the OpenEnv spec.

---

## 💡 Design Decisions

**Why data pipelines?**
Every company has them. Debugging broken pipelines is a genuine, high-value
engineering task. There are no existing OpenEnv environments in this domain.

**Why pandas?**
Pandas is the industry standard for data transformation in Python. The agent
reads and writes real Python/pandas code — not a toy DSL.

**Why partial rewards?**
Binary rewards (0 or 1 at episode end) provide almost no learning signal.
Our reward function gives the agent feedback at every step: did running the
pipeline help? Did the fix clean up an error? This makes the environment
suitable for both LLM prompting and RL training.

**Why cascading bugs in the hard task?**
Real pipelines fail in cascades. A schema change in stage 1 silently breaks
stage 3. This tests whether agents understand data *flow*, not just isolated bugs.