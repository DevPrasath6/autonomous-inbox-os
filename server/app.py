import sys
import os

try:
    from ..models import EmailAction, EmailObservation, EmailState, StepResult, TaskDefinition
    from ..server.environment import EmailEnvironment, TASKS
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import EmailAction, EmailObservation, EmailState, StepResult, TaskDefinition
    from server.environment import EmailEnvironment, TASKS

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict, Any
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


@app.get("/")
def root():
    return JSONResponse({
        "name": "Autonomous Inbox OS",
        "version": "1.0.0",
        "status": "running",
        "openenv": True,
        "endpoints": [
            "/reset", "/step", "/state", "/tasks", "/health", "/metrics", "/grader",
            "/metadata", "/schema", "/mcp"
        ],
    })


@app.get("/health")
def health():
    return JSONResponse({"status": "healthy"})


@app.get("/metadata")
def metadata():
    return JSONResponse({
        "name": "autonomous_inbox_os",
        "description": "OpenEnv-compliant AI email triage environment",
        "version": "1.0.0",
    })


@app.get("/schema")
def schema():
    return JSONResponse({
        "action": EmailAction.model_json_schema(),
        "observation": EmailObservation.model_json_schema(),
        "state": EmailState.model_json_schema(),
    })


@app.post("/mcp")
async def mcp(request: Request):
    payload = await request.json()
    method = payload.get("method")
    request_id = payload.get("id")

    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "autonomous_inbox_os", "version": "1.0.0"},
                "capabilities": {},
            },
        })

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    })


@app.get("/demo", response_class=FileResponse)
def demo():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.api_route("/reset", methods=["GET", "POST", "PUT", "OPTIONS"])
async def reset(request: Request):
    global _env, _metrics_log
    task_id = 2
    try:
        body = await request.body()
        if body and len(body) > 0:
            import json
            data = json.loads(body)
            if isinstance(data, dict):
                task_id = int(data.get("task_id", 2))
    except Exception:
        pass
    try:
        qp = request.query_params.get("task_id")
        if qp:
            task_id = int(qp)
    except Exception:
        pass
    task_id = max(1, min(3, task_id))
    _env = EmailEnvironment(task_id=task_id)
    _metrics_log = []
    obs = _env.reset()
    return JSONResponse(status_code=200, content={
        "observation": {
            "email_id": obs.email_id,
            "subject": obs.subject,
            "body": obs.body,
            "sender": obs.sender,
            "has_attachment": obs.has_attachment,
            "meeting_request": obs.meeting_request,
            "inbox_size": obs.inbox_size,
            "pending_urgent": obs.pending_urgent,
            "pending_important": obs.pending_important,
            "user_stress_level": obs.user_stress_level,
            "time_remaining": obs.time_remaining,
            "step_count": obs.step_count,
        },
        "reward": 0.0,
        "done": False,
        "info": {"task_id": task_id}
    })


@app.api_route("/step", methods=["POST", "PUT"])
async def step(request: Request):
    global _metrics_log
    try:
        body = await request.json()
        action = EmailAction(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        result = _env.step(action)
        _metrics_log.append({
            "step": _env._step_count,
            "reward": result.reward.total,
            "stress": _env._stress,
            "action": action.action_type.value,
        })
        obs_data = None
        if result.observation:
            obs_data = result.observation.model_dump()
        return JSONResponse(content={
            "observation": obs_data,
            "reward": result.reward.total,
            "done": result.done,
            "info": result.info,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return JSONResponse(content=_env.state().model_dump())


@app.get("/tasks")
def get_tasks():
    return JSONResponse(content=[t.model_dump() for t in TASKS])


@app.get("/grader")
def grader():
    return JSONResponse(content=_env.run_grader())


@app.get("/metrics")
def metrics():
    s = _env.state()
    return JSONResponse(content={
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
    })


@app.api_route("/simulate", methods=["GET", "POST"])
async def simulate(request: Request):
    import random
    from models import ActionType, Priority
    task_id = 2
    policy = "random"
    try:
        task_id = int(request.query_params.get("task_id", 2))
        policy = request.query_params.get("policy", "random")
    except Exception:
        pass
    env = EmailEnvironment(task_id=task_id)
    env.reset()
    total_reward = 0.0
    steps = 0
    actions_map = list(ActionType)
    labels_map = list(Priority)
    while not env._done:
        if policy == "random":
            action = EmailAction(action_type=random.choice(actions_map), classification=random.choice(labels_map))
        else:
            action = EmailAction(action_type=ActionType.archive, classification=Priority.spam)
        result = env.step(action)
        total_reward += result.reward.total
        steps += 1
        if result.done:
            break
    grader_result = env.run_grader()
    return JSONResponse(content={
        "policy": policy,
        "task_id": task_id,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "grader_score": grader_result["score"],
        "details": grader_result["details"],
    })


def main():
    """Entry point for openenv-core server discovery."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()
