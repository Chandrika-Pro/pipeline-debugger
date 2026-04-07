"""
server/app.py — Pipeline Debugger OpenEnv Server
Entry point for openenv validate multi-mode deployment
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from typing import Optional, List, Dict
import uvicorn
import sys
import os

# Add parent directory to path so we can import environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    <html><head><title>Pipeline Debugger — OpenEnv</title></head>
    <body style="font-family:sans-serif;max-width:750px;margin:50px auto;padding:20px;background:#0f1117;color:#e0e0e0">
        <h1 style="color:#4a9eff">🔧 Pipeline Debugger — OpenEnv</h1>
        <p>A real-world environment where AI agents debug broken data pipelines.</p>
        <h2 style="color:#4a9eff;margin-top:20px">Tasks</h2>
        <ul>
            <li><b>easy</b> — Type cast bug in sales pipeline</li>
            <li><b>medium</b> — Silent date filter bug in monthly report</li>
            <li><b>hard</b> — 3 cascading bugs in logistics pipeline</li>
        </ul>
        <h2 style="color:#4a9eff;margin-top:20px">API</h2>
        <ul>
            <li><code>GET  /health</code></li>
            <li><code>GET  /tasks</code></li>
            <li><code>POST /reset</code></li>
            <li><code>POST /step</code></li>
            <li><code>GET  /state?task=easy</code></li>
        </ul>
        <p><a href="/docs" style="color:#4a9eff">📖 Swagger UI</a></p>
    </body></html>
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
        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "openenv.yaml"
        )
        with open(yaml_path, "r") as f:
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


def main():
    """Main entry point for openenv validate."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()