"""
app.py — FastAPI server for Pipeline Debugger OpenEnv
Runs on port 7860 for Hugging Face Spaces

Endpoints:
  GET  /           → homepage
  GET  /health     → health check (validator pings this)
  GET  /tasks      → list all tasks
  GET  /openenv.yaml → serve spec file
  POST /reset      → start episode
  POST /step       → take action
  GET  /state      → current state
"""

from fastapi import FastAPI, HTTPException
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


class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    task: str = "easy"
    action_type: str
    stage_id: Optional[str] = None
    new_code: Optional[str] = None
    column_name: Optional[str] = None
    new_type: Optional[str] = None
    new_condition: Optional[str] = None
    new_order: Optional[List[str]] = None
    reasoning: Optional[str] = None


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
    <body style="font-family:sans-serif;max-width:750px;margin:50px auto;padding:20px">
        <h1>🔧 Pipeline Debugger — OpenEnv</h1>
        <p>A real-world environment where AI agents debug broken data pipelines.</p>
        <h2>Tasks</h2>
        <ul>
            <li><b>easy</b> — Type cast bug in sales pipeline</li>
            <li><b>medium</b> — Silent date filter bug in monthly report</li>
            <li><b>hard</b> — 3 cascading bugs in logistics pipeline</li>
        </ul>
        <h2>API</h2>
        <ul>
            <li><code>GET  /health</code></li>
            <li><code>GET  /tasks</code></li>
            <li><code>POST /reset</code></li>
            <li><code>POST /step</code></li>
            <li><code>GET  /state?task=easy</code></li>
        </ul>
        <p><a href="/docs">📖 Swagger UI</a></p>
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
            {"id": "easy_type_mismatch",     "name": "easy",   "difficulty": "easy",   "max_steps": 15},
            {"id": "medium_silent_filter_bug","name": "medium", "difficulty": "medium", "max_steps": 20},
            {"id": "hard_cascade_bugs",       "name": "hard",   "difficulty": "hard",   "max_steps": 30},
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
def reset(req: ResetRequest):
    try:
        env = get_env(req.task)
        obs = env.reset()
        return {"observation": obs_to_dict(obs)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    try:
        env = get_env(req.task)
        action = Action(
            action_type=req.action_type,
            stage_id=req.stage_id,
            new_code=req.new_code,
            column_name=req.column_name,
            new_type=req.new_type,
            new_condition=req.new_condition,
            new_order=req.new_order,
            reasoning=req.reasoning,
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