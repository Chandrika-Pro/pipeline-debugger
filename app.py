"""
app.py — FastAPI server for Pipeline Debugger OpenEnv
Runs on port 7860 for Hugging Face Spaces
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn

from environment.env import PipelineDebuggerEnv
from environment.models import Action

app = FastAPI(
    title="Pipeline Debugger — OpenEnv",
    description="AI agents debug broken real-world data pipelines.",
    version="1.0.0",
)

_envs: Dict[str, PipelineDebuggerEnv] = {}


def get_env(task: str) -> PipelineDebuggerEnv:
    if task not in ["easy", "medium", "hard"]:
        raise ValueError(f"Unknown task '{task}'. Choose: easy, medium, hard")
    if task not in _envs:
        _envs[task] = PipelineDebuggerEnv(task)
    return _envs[task]


def obs_to_dict(obs) -> dict:
    return {
        "task_id": obs.task_id,
        "task_description": obs.task_description,
        "pipeline_stages": [
            {
                "stage_id": s.stage_id,
                "stage_type": s.stage_type,
                "description": s.description,
                "code": s.code,
                "inputs": s.inputs,
                "status": s.status,
                "error_message": s.error_message,
            }
            for s in obs.pipeline_stages
        ],
        "input_data_sample": obs.input_data_sample,
        "current_output_sample": obs.current_output_sample,
        "expected_output_sample": obs.expected_output_sample,
        "expected_schema": obs.expected_schema,
        "logs": obs.logs,
        "actions_taken": obs.actions_taken,
        "steps_remaining": obs.steps_remaining,
        "current_score": obs.current_score,
    }


def reward_to_dict(r) -> dict:
    return {
        "value": r.value,
        "schema_match": r.schema_match,
        "data_match": r.data_match,
        "progress": r.progress,
        "efficiency_penalty": r.efficiency_penalty,
        "crash_penalty": r.crash_penalty,
        "reason": r.reason,
    }


def state_to_dict(s) -> dict:
    return {
        "task_id": s.task_id,
        "step_count": s.step_count,
        "done": s.done,
        "total_reward": s.total_reward,
        "pipeline_fixed": s.pipeline_fixed,
        "bugs_fixed": s.bugs_fixed,
        "bugs_remaining": s.bugs_remaining,
    }


@app.get("/", response_class=HTMLResponse)
def homepage():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Debugger — OpenEnv</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; }
        .header {
            background: linear-gradient(135deg, #1a1f2e, #2d3561);
            padding: 50px 20px;
            text-align: center;
            border-bottom: 2px solid #4a9eff;
        }
        .header h1 { font-size: 2.8em; color: #4a9eff; margin-bottom: 12px; }
        .header p { font-size: 1.1em; color: #a0a0b0; max-width: 600px; margin: 0 auto 20px; }
        .badge {
            display: inline-block; background: #4a9eff22;
            border: 1px solid #4a9eff; color: #4a9eff;
            padding: 4px 14px; border-radius: 20px; font-size: 0.8em; margin: 4px;
        }
        .status {
            display: inline-flex; align-items: center; gap: 8px;
            background: #1a3a1a; border: 1px solid #4caf50; color: #4caf50;
            padding: 8px 20px; border-radius: 20px; font-size: 0.9em; margin-top: 16px;
        }
        .dot {
            width: 8px; height: 8px; background: #4caf50;
            border-radius: 50%; animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        .container { max-width: 1000px; margin: 0 auto; padding: 40px 20px; }
        .section { margin-bottom: 50px; }
        .section h2 {
            font-size: 1.4em; color: #4a9eff; margin-bottom: 20px;
            padding-bottom: 10px; border-bottom: 1px solid #2a2f3e;
        }
        .scores-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center; margin-bottom: 20px; }
        .score-card {
            background: #1a1f2e; border: 1px solid #2a2f3e;
            border-radius: 12px; padding: 24px; transition: border-color 0.3s;
        }
        .score-card:hover { border-color: #4a9eff; }
        .score { font-size: 2.2em; font-weight: bold; color: #4a9eff; }
        .score-label { font-size: 0.9em; color: #a0a0b0; margin-top: 6px; }
        .score-steps { font-size: 0.8em; margin-top: 6px; }
        .avg-row {
            text-align: center; background: #1a1f2e;
            border: 1px solid #4a9eff33; border-radius: 12px; padding: 16px; margin-top: 10px;
        }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
        .card {
            background: #1a1f2e; border: 1px solid #2a2f3e;
            border-radius: 12px; padding: 24px; transition: border-color 0.3s;
        }
        .card:hover { border-color: #4a9eff; }
        .card h3 { font-size: 1.05em; margin-bottom: 10px; color: #e0e0e0; }
        .card p { color: #808090; font-size: 0.88em; line-height: 1.7; }
        .difficulty {
            display: inline-block; padding: 3px 12px; border-radius: 12px;
            font-size: 0.75em; font-weight: bold; margin-bottom: 12px;
        }
        .easy   { background: #1a3a1a; color: #4caf50; border: 1px solid #4caf50; }
        .medium { background: #3a2a1a; color: #ff9800; border: 1px solid #ff9800; }
        .hard   { background: #3a1a1a; color: #f44336; border: 1px solid #f44336; }
        .reward-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .reward-card { background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 12px; padding: 20px; }
        .reward-card h3 { font-size: 1em; margin-bottom: 12px; }
        .reward-card p { color: #808090; font-size: 0.88em; line-height: 2; }
        .api-list { list-style: none; }
        .api-list li {
            background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 8px;
            padding: 12px 18px; margin-bottom: 10px; font-family: monospace;
            font-size: 0.9em; display: flex; align-items: center; gap: 12px; transition: border-color 0.3s;
        }
        .api-list li:hover { border-color: #4a9eff; }
        .method { padding: 3px 10px; border-radius: 5px; font-size: 0.8em; font-weight: bold; min-width: 50px; text-align: center; }
        .get  { background: #1a3a2a; color: #4caf50; }
        .post { background: #1a2a3a; color: #4a9eff; }
        .swagger-btn {
            display: inline-block; margin-top: 24px; background: #4a9eff; color: white;
            padding: 12px 30px; border-radius: 8px; text-decoration: none;
            font-weight: bold; font-size: 0.95em; transition: background 0.3s;
        }
        .swagger-btn:hover { background: #3a8eef; }
        .footer {
            text-align: center; padding: 30px; color: #404050;
            border-top: 1px solid #2a2f3e; margin-top: 20px; font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔧 Pipeline Debugger</h1>
        <p>A real-world OpenEnv environment where AI agents debug broken data pipelines</p>
        <div style="margin: 14px 0;">
            <span class="badge">OpenEnv</span>
            <span class="badge">Data Engineering</span>
            <span class="badge">Real-World</span>
            <span class="badge">pandas</span>
            <span class="badge">FastAPI</span>
        </div>
        <div>
            <span class="status"><span class="dot"></span> Running on Hugging Face Spaces</span>
        </div>
    </div>

    <div class="container">
        <div class="section">
            <h2>📊 Baseline Scores &nbsp;<small style="color:#606070;font-size:0.7em;">model: meta-llama/Llama-3.3-70B-Instruct</small></h2>
            <div class="scores-grid">
                <div class="score-card">
                    <div class="score">1.00</div>
                    <div class="score-label">Easy Task</div>
                    <div class="score-steps" style="color:#4caf50;">✓ 5 steps | perfect</div>
                </div>
                <div class="score-card">
                    <div class="score">1.00</div>
                    <div class="score-label">Medium Task</div>
                    <div class="score-steps" style="color:#ff9800;">✓ 6 steps | perfect</div>
                </div>
                <div class="score-card">
                    <div class="score">0.93</div>
                    <div class="score-label">Hard Task</div>
                    <div class="score-steps" style="color:#f44336;">✓ 6 steps | near perfect</div>
                </div>
            </div>
            <div class="avg-row">
                <span style="color:#808090;">Average Score: </span>
                <span style="color:#4a9eff; font-size:1.5em; font-weight:bold;">0.9778</span>
                &nbsp;&nbsp;
                <span style="color:#808090;">Total Time: </span>
                <span style="color:#a0a0b0;">22.5 seconds</span>
            </div>
        </div>

        <div class="section">
            <h2>🎯 Tasks</h2>
            <div class="cards">
                <div class="card">
                    <span class="difficulty easy">EASY</span>
                    <h3>Type Mismatch Bug</h3>
                    <p>A retail shop's sales pipeline calculates wrong revenue because prices are cast to int instead of float. Agent must find and fix the type cast in stage_2.</p>
                </div>
                <div class="card">
                    <span class="difficulty medium">MEDIUM</span>
                    <h3>Silent Filter Bug</h3>
                    <p>Monthly report pipeline runs with no errors but produces wrong output. A date boundary filter silently includes records from the wrong month.</p>
                </div>
                <div class="card">
                    <span class="difficulty hard">HARD</span>
                    <h3>Cascading Bugs</h3>
                    <p>3 bugs hidden across 4 stages in a logistics pipeline. Each bug masks the next. Agent must fix them in the correct order.</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>💰 Reward Function</h2>
            <div class="reward-cards">
                <div class="reward-card">
                    <h3 style="color:#4caf50;">✅ Positive Rewards</h3>
                    <p>+0.30 schema correct<br>+0.50 data values match<br>+0.20 bugs confirmed fixed<br>+0.05 pipeline runs clean<br>+0.02 valid fix applied</p>
                </div>
                <div class="reward-card">
                    <h3 style="color:#f44336;">❌ Penalties</h3>
                    <p>-0.10 pipeline crash<br>-0.05 invalid action<br>-0.02 too many runs<br>-0.02 stage not found</p>
                </div>
                <div class="reward-card" style="text-align:center;">
                    <h3 style="color:#4a9eff;">🏆 Max Score</h3>
                    <div style="font-size:3em; color:#4a9eff; font-weight:bold; margin:16px 0;">1.0</div>
                    <p style="color:#606070;">per episode<br>partial credit<br>at every step</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🌐 API Endpoints</h2>
            <ul class="api-list">
                <li><span class="method get">GET</span> /health — Health check</li>
                <li><span class="method get">GET</span> /tasks — List all tasks</li>
                <li><span class="method post">POST</span> /reset — Start a new episode</li>
                <li><span class="method post">POST</span> /step — Take an action</li>
                <li><span class="method get">GET</span> /state — Current state snapshot</li>
                <li><span class="method get">GET</span> /openenv.yaml — OpenEnv spec metadata</li>
            </ul>
            <a href="/docs" class="swagger-btn">📖 Open Swagger UI</a>
        </div>
    </div>

    <div class="footer">
        🔧 Pipeline Debugger — OpenEnv &nbsp;|&nbsp;
        Built for Scaler School of Technology Hackathon &nbsp;|&nbsp;
        <a href="/docs" style="color:#4a9eff; text-decoration:none;">API Docs</a>
    </div>
</body>
</html>
    """


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "pipeline-debugger",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "easy_type_mismatch",      "name": "easy",   "difficulty": "easy",   "max_steps": 15},
            {"id": "medium_silent_filter_bug", "name": "medium", "difficulty": "medium", "max_steps": 20},
            {"id": "hard_cascade_bugs",        "name": "hard",   "difficulty": "hard",   "max_steps": 30},
        ]
    }


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def serve_openenv_yaml():
    try:
        with open("openenv.yaml", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


@app.post("/reset")
async def reset(request: Request):
    """Start a fresh episode. Accepts empty body or {task: 'easy'}"""
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        task = body.get("task", "easy") if body else "easy"
        env = get_env(task)
        obs = env.reset()
        return {"observation": obs_to_dict(obs)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: Request):
    """Take one action."""
    try:
        body = await request.json()
        task = body.get("task", "easy")
        env = get_env(task)
        action = Action(
            action_type=body.get("action_type"),
            stage_id=body.get("stage_id"),
            new_code=body.get("new_code"),
            column_name=body.get("column_name"),
            new_type=body.get("new_type"),
            new_condition=body.get("new_condition"),
            new_order=body.get("new_order"),
            reasoning=body.get("reasoning"),
        )
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs_to_dict(obs),
            "reward": reward_to_dict(reward),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state(task: str = "easy"):
    """Current state snapshot."""
    try:
        env = get_env(task)
        s = env.state()
        return {"state": state_to_dict(s)}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)