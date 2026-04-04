import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from models import EmailAction, EmailObservation, EmailState, StepResult, TaskDefinition
from server.environment import EmailEnvironment, TASKS
from typing import List, Dict, Any, Optional
import time

app = FastAPI(
    title="Autonomous Inbox OS",
    description="OpenEnv-compliant AI Email Triage Environment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

_env = EmailEnvironment(task_id=2)
_start_time = time.time()
_metrics_log: List[Dict] = []


def _do_reset(task_id: int = 2):
    global _env, _metrics_log
    _env = EmailEnvironment(task_id=task_id)
    _metrics_log = []
    obs = _env.reset()
    return JSONResponse(content=obs.model_dump())


@app.get("/")
def root():
    return JSONResponse({
        "name": "Autonomous Inbox OS",
        "version": "1.0.0",
        "status": "running",
        "openenv": True,
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/metrics", "/simulate", "/grader", "/demo"],
    })


@app.get("/demo", response_class=FileResponse)
def demo():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.post("/reset")
async def reset(request: Request):
    """POST /reset — OpenEnv spec. Accepts empty body or JSON with task_id."""
    task_id = 2
    try:
        body = await request.json()
        if isinstance(body, dict):
            task_id = int(body.get("task_id", 2))
    except Exception:
        pass
    qp = request.query_params.get("task_id")
    if qp:
        try:
            task_id = int(qp)
        except Exception:
            pass
    task_id = max(1, min(3, task_id))
    return _do_reset(task_id)


@app.get("/reset")
def reset_get(task_id: int = Query(default=2, ge=1, le=3)):
    return _do_reset(task_id)


@app.post("/step")
def step(action: EmailAction) -> StepResult:
    try:
        result = _env.step(action)
        _metrics_log.append({
            "step": _env._step_count,
            "reward": result.reward.total,
            "stress": _env._stress,
            "action": action.action_type.value,
        })
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state() -> EmailState:
    return _env.state()


@app.get("/tasks")
def get_tasks() -> List[TaskDefinition]:
    return TASKS


@app.get("/grader")
def grader() -> Dict[str, Any]:
    return _env.run_grader()


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    s = _env.state()
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "inbox_size": s.inbox_size,
        "processed": s.processed,
        "accuracy": round(s.correct_classifications / max(1, s.processed), 4),
        "missed_urgent": s.missed_urgent,
        "user_stress_level": s.user_stress_level,
        "time_remaining": s.time_remaining,
        "cumulative_reward": s.cumulative_reward,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "recent_steps": _metrics_log[-10:] if _metrics_log else [],
    }


@app.post("/simulate")
def simulate(
    task_id: int = Query(default=2, ge=1, le=3),
    policy: str = Query(default="random")
) -> Dict[str, Any]:
    import random
    from models import ActionType, Priority

    env = EmailEnvironment(task_id=task_id)
    env.reset()
    total_reward = 0.0
    steps = 0
    actions_map = list(ActionType)
    labels_map = list(Priority)

    while not env._done:
        if policy == "random":
            action = EmailAction(action_type=random.choice(actions_map), classification=random.choice(labels_map))
        elif policy == "always_important":
            action = EmailAction(action_type=ActionType.reply, classification=Priority.important)
        else:
            action = EmailAction(action_type=ActionType.archive, classification=Priority.spam)
        result = env.step(action)
        total_reward += result.reward.total
        steps += 1
        if result.done:
            break

    grader_result = env.run_grader()
    return {
        "policy": policy,
        "task_id": task_id,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "grader_score": grader_result["score"],
        "details": grader_result["details"],
    }
