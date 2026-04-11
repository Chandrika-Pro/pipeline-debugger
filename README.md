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

# 🔧 Pipeline Debugger — OpenEnv

> A real-world OpenEnv environment where AI agents debug broken data pipelines.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-4a9eff)](https://huggingface.co/spaces/Chandrika23/pipeline-debugger)
[![HF Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Chandrika23/pipeline-debugger)
[![GitHub](https://img.shields.io/badge/GitHub-repo-black)](https://github.com/Chandrika-Pro/pipeline-debugger)

---

## 🧠 Motivation

Every company runs **data pipelines** — automated sequences that clean, transform, and aggregate raw data into useful reports. When these pipelines break, engineers spend hours tracking down bugs. This is a genuine, high-value engineering task worth billions in lost productivity every year.

This environment simulates exactly that problem. An AI agent is given a **broken pipeline** and must:
1. **Inspect** stages to understand what's going wrong
2. **Fix** the broken code
3. **Verify** by running the pipeline
4. **Submit** when confident the output matches expected

The environment provides **partial rewards throughout the episode** — not just binary pass/fail — making it suitable for both LLM prompting and reinforcement learning.

---

## 🗂 Project Structure

```
openenv-pipeline-debugger/
├── inference.py              ← Baseline agent script (root, mandatory)
├── validate.py               ← Pre-submission validator (50 checks)
├── app.py                    ← FastAPI server
├── openenv.yaml              ← OpenEnv spec metadata
├── Dockerfile                ← Container definition
├── requirements.txt          ← Python dependencies
├── pyproject.toml            ← Package config for openenv validate
├── server/
│   └── app.py                ← Server entry point for multi-mode deploy
├── environment/
│   ├── env.py                ← Main environment (reset/step/state)
│   ├── models.py             ← Typed dataclass models
│   ├── pipeline.py           ← Pipeline execution engine
│   ├── graders.py            ← Scoring logic (0.0–1.0)
│   └── tasks/
│       ├── task_easy.py      ← Type mismatch bug
│       ├── task_medium.py    ← Silent filter bug
│       └── task_hard.py      ← 3 cascading bugs
└── tests/
    └── test_env.py           ← 13 unit tests
```

---

## 🔁 OpenEnv Interface

```python
from environment.env import PipelineDebuggerEnv
from environment.models import Action

# Start a new episode
env = PipelineDebuggerEnv(task_name="easy")  # easy | medium | hard
obs = env.reset()

# Agent inspects a stage
obs, reward, done, info = env.step(Action(
    action_type="inspect_stage",
    stage_id="stage_2"
))

# Agent fixes the bug
obs, reward, done, info = env.step(Action(
    action_type="fix_transform",
    stage_id="stage_2",
    new_code="df['price'] = df['price'].astype(float)\nresult = df",
    reasoning="price should be float not int"
))

# Submit for grading
obs, reward, done, info = env.step(Action(action_type="submit"))
print(f"Final score: {info['grader_breakdown']['final_score']}")

# Get state snapshot
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
| `expected_schema` | Dict[str, str] | Column names and expected types |
| `logs` | List[str] | Pipeline run logs and error messages |
| `actions_taken` | List[str] | History of agent actions this episode |
| `steps_remaining` | int | Steps left before episode ends |
| `current_score` | float | Cumulative reward so far |

Each `StageInfo` contains `stage_id`, `stage_type`, `description`, `code`, `status`, `error_message`.

---

## 🕹 Action Space

| `action_type` | Required Fields | Description |
|---|---|---|
| `inspect_stage` | `stage_id` | Reveal full details of a stage |
| `fix_transform` | `stage_id`, `new_code` | Replace a stage's transformation code |
| `fix_schema` | `stage_id`, `column_name`, `new_type` | Add a type cast to a stage |
| `fix_filter` | `stage_id`, `new_condition` | Replace a filter condition |
| `reorder_stages` | `new_order` | Change stage execution order |
| `run_pipeline` | — | Execute pipeline and observe output |
| `submit` | — | Finalize fix and trigger grading |

All actions accept an optional `reasoning` string for chain-of-thought logging.

---

## 💰 Reward Function

Rewards are given at **every step** — not just at the end:

| Event | Reward |
|---|---|
| Schema correct on submit | up to +0.30 |
| Data values correct on submit | up to +0.50 |
| Bugs confirmed fixed | up to +0.20 |
| Pipeline runs clean after fix | +0.05 |
| Valid fix applied | +0.02 |
| Pipeline crashes | **-0.10** |
| Too many run_pipeline calls | -0.02 each |
| Invalid action | -0.05 |
| **Maximum score per episode** | **1.0** |

---

## 📋 Tasks

### Task 1 — Easy: Type Mismatch (`easy_type_mismatch`)

**Domain:** Retail sales pipeline

**Story:** A shop's pipeline calculates total revenue per product. Prices like ₹599.99 are being truncated to ₹599 because one stage casts the price column to `int` instead of `float`.

**Bug:** `stage_2` — `df['price'].astype(int)` should be `astype(float)`

**Max steps:** 15 | **Expected score:** ~0.85–1.0 | **Difficulty:** ⭐

---

### Task 2 — Medium: Silent Filter Bug (`medium_silent_filter_bug`)

**Domain:** E-commerce monthly reporting

**Story:** A monthly orders report shows wrong totals. The pipeline runs with **NO errors** — no crashes, no exceptions. But the output includes one extra order from January because the date boundary filter uses the wrong start date.

**Bug:** `stage_3` — filter start is `2024-01-31` should be `2024-02-01`

**Max steps:** 20 | **Expected score:** ~0.55–0.70 | **Difficulty:** ⭐⭐⭐

---

### Task 3 — Hard: Cascading Bugs (`hard_cascade_bugs`)

**Domain:** Logistics shipment reporting

**Story:** A logistics pipeline has THREE bugs, each masking the next:
1. `stage_2` renames `warehouse_id` → `wh_id`, breaking the downstream join
2. `stage_3` join uses wrong key, producing NULL regions
3. `stage_4` classifies shipments with threshold `> 500 km` instead of `> 1000 km`

Fixing bug 3 first does nothing if bugs 1 and 2 still corrupt the data. Agent must reason about data flow and fix in correct order.

**Max steps:** 30 | **Expected score:** ~0.25–0.45 | **Difficulty:** ⭐⭐⭐⭐⭐

---

## 📊 Baseline Scores

Tested with `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace router (temperature=0.1):

| Task | Score | Steps Used | Time |
|---|---|---|---|
| easy | **1.0000** | 5 | ~7s |
| medium | **1.0000** | 6 | ~8s |
| hard | **0.9333** | 6 | ~8s |
| **Average** | **0.9778** | — | 22.5s total |

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone the repo
git clone https://github.com/Chandrika-Pro/pipeline-debugger
cd pipeline-debugger

# Install dependencies
pip install -r requirements.txt

# Run validator (must show 50/50)
python validate.py

# Run tests (must show 13/13)
python tests/test_env.py

# Start the server
python app.py
# → http://localhost:7860
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_hf_token"

python inference.py
```

### Docker

```bash
docker build -t pipeline-debugger .
docker run -p 7860:7860 pipeline-debugger
```

### HTTP API Quick Test

```bash
# Health check
curl http://localhost:7860/health

# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'

# Take action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "action_type": "run_pipeline"}'
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

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take an action |
| GET | `/state` | Current state snapshot |
| GET | `/openenv.yaml` | OpenEnv spec |
| GET | `/docs` | Swagger UI |

---

## 💡 Design Decisions

**Why data pipelines?**
Every company has them. Debugging broken pipelines is a genuine, high-value engineering task. There are no existing OpenEnv environments in this domain.

**Why pandas?**
Pandas is the industry standard for data transformation in Python. The agent reads and writes real Python/pandas code — not a toy DSL.

**Why partial rewards?**
Binary rewards provide almost no learning signal. Our reward function gives the agent feedback at every step — did the fix clean up an error? Did the pipeline run clean? This makes the environment suitable for both LLM prompting and RL training.

**Why cascading bugs in the hard task?**
Real pipelines fail in cascades. A schema change in stage 1 silently breaks stage 3. This tests whether agents understand data *flow*, not just isolated bugs.

---

## 📁 Links

- 🤗 **Live Demo:** https://huggingface.co/spaces/Chandrika23/pipeline-debugger
- 💻 **GitHub:** https://github.com/Chandrika-Pro/pipeline-debugger
- 📖 **API Docs:** https://Chandrika23-pipeline-debugger.hf.space/docs