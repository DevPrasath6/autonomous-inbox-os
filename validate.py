#!/usr/bin/env python3
"""
validate.py — Pre-submission validator for Autonomous Inbox OS.

Runs all OpenEnv spec checks locally before you submit.
All checks must pass or you risk disqualification.

Usage:
    python validate.py
    python validate.py --verbose
"""

import sys
import os
import json
import importlib
import argparse
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

def check(name, fn, critical=True):
    try:
        msg = fn()
        results.append((PASS, name, msg or ""))
        print(f"  {PASS}  {name}" + (f" — {msg}" if msg else ""))
        return True
    except Exception as e:
        results.append((FAIL if critical else WARN, name, str(e)))
        icon = FAIL if critical else WARN
        print(f"  {icon}  {name}")
        print(f"       {e}")
        if args.verbose:
            traceback.print_exc()
        return False


# ── 1. FILE STRUCTURE ─────────────────────────────────────────────────────────
def check_files():
    print("\n📁 File structure")
    required = [
        "models.py",
        "inference.py",
        "openenv.yaml",
        "Dockerfile",
        "README.md",
        "data/emails.json",
        "server/environment.py",
        "server/app.py",
        "server/requirements.txt",
    ]
    missing = [f for f in required if not (ROOT / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    return f"{len(required)} required files present"

def check_inference_at_root():
    assert (ROOT / "inference.py").exists(), "inference.py must be in root directory"
    return "inference.py at project root"

def check_openenv_yaml():
    import yaml
    with open(ROOT / "openenv.yaml") as f:
        data = yaml.safe_load(f)
    required_keys = ["name", "version", "description", "tasks", "observation_space", "action_space"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"openenv.yaml missing keys: {missing}")
    task_count = len(data.get("tasks", []))
    if task_count < 3:
        raise ValueError(f"Need ≥3 tasks in openenv.yaml, found {task_count}")
    return f"{task_count} tasks defined"


# ── 2. MODELS ─────────────────────────────────────────────────────────────────
def check_models():
    print("\n🧩 Pydantic models")

def check_models_importable():
    from models import EmailAction, EmailObservation, EmailState, EmailReward
    return "EmailAction, EmailObservation, EmailState, EmailReward"

def check_action_model():
    from models import EmailAction, ActionType, Priority
    a = EmailAction(action_type=ActionType.archive, classification=Priority.spam)
    assert a.action_type == ActionType.archive
    return "EmailAction instantiates correctly"

def check_observation_model():
    from models import EmailObservation
    o = EmailObservation(
        email_id=1, subject="Test", body="Body", sender="a@b.com",
        inbox_size=5, user_stress_level=0.3, time_remaining=10
    )
    assert 0.0 <= o.user_stress_level <= 1.0
    return "EmailObservation with stress field [0,1]"


# ── 3. ENVIRONMENT ────────────────────────────────────────────────────────────
def check_environment():
    print("\n⚙️  Environment spec")

def check_env_importable():
    from server.environment import EmailEnvironment, TASKS
    return f"EmailEnvironment, {len(TASKS)} tasks"

def check_reset():
    from server.environment import EmailEnvironment
    env = EmailEnvironment(task_id=1)
    obs = env.reset()
    assert obs is not None
    assert obs.email_id is not None
    assert obs.subject
    assert obs.user_stress_level == 0.0, "reset() must clear stress to 0"
    return f"reset() → clean state, first email: '{obs.subject[:30]}…'"

def check_step():
    from server.environment import EmailEnvironment
    from models import EmailAction, ActionType, Priority
    env = EmailEnvironment(task_id=1)
    env.reset()
    action = EmailAction(action_type=ActionType.archive, classification=Priority.spam, reasoning="Test")
    result = env.step(action)
    assert result.observation is not None or result.done
    assert isinstance(result.reward.total, float)
    assert isinstance(result.done, bool)
    return f"step() → reward={result.reward.total}, done={result.done}"

def check_state():
    from server.environment import EmailEnvironment
    env = EmailEnvironment(task_id=1)
    env.reset()
    s = env.state()
    assert s.episode_id
    assert s.step_count == 0
    assert s.inbox_size > 0
    return f"state() → episode_id={s.episode_id[:8]}…, inbox_size={s.inbox_size}"

def check_episode_completes():
    from server.environment import EmailEnvironment
    from models import EmailAction, ActionType, Priority
    import random
    env = EmailEnvironment(task_id=1)
    env.reset()
    steps = 0
    while not env._done and steps < 200:
        a = EmailAction(
            action_type=random.choice(list(ActionType)),
            classification=random.choice(list(Priority)),
        )
        r = env.step(a)
        steps += 1
        if r.done:
            break
    assert env._done or steps < 200, "Episode never terminated"
    return f"Episode completed in {steps} steps"


# ── 4. TASKS & GRADERS ───────────────────────────────────────────────────────
def check_tasks_section():
    print("\n🎯 Tasks & graders")

def check_three_tasks():
    from server.environment import TASKS
    assert len(TASKS) >= 3, f"Need ≥3 tasks, got {len(TASKS)}"
    difficulties = {t.difficulty for t in TASKS}
    assert "easy" in difficulties and "hard" in difficulties, \
        f"Need easy→hard range, got: {difficulties}"
    return f"{len(TASKS)} tasks: {[t.difficulty for t in TASKS]}"

def check_grader_scores():
    from server.environment import EmailEnvironment
    from models import EmailAction, ActionType, Priority
    import random

    scores = []
    for task_id in [1, 2, 3]:
        env = EmailEnvironment(task_id=task_id)
        env.reset()
        while not env._done:
            a = EmailAction(
                action_type=random.choice(list(ActionType)),
                classification=random.choice(list(Priority)),
            )
            r = env.step(a)
            if r.done:
                break
        result = env.run_grader()
        score = result["score"]
        assert 0.0 <= score <= 1.0, f"Task {task_id} score {score} outside [0,1]"
        scores.append(score)

    return f"Task scores: {[round(s,3) for s in scores]} — all in [0.0, 1.0]"

def check_grader_deterministic():
    from server.environment import EmailEnvironment
    from models import EmailAction, ActionType, Priority

    def run_fixed():
        env = EmailEnvironment(task_id=2)
        env.reset()
        while not env._done:
            a = EmailAction(action_type=ActionType.archive, classification=Priority.spam)
            r = env.step(a)
            if r.done:
                break
        return env.run_grader()["score"]

    s1, s2 = run_fixed(), run_fixed()
    assert s1 == s2, f"Grader not deterministic: {s1} ≠ {s2}"
    return f"Deterministic: same fixed policy → score={s1}"


# ── 5. REWARD FUNCTION ───────────────────────────────────────────────────────
def check_reward_section():
    print("\n💰 Reward function")

def check_reward_is_dense():
    from server.environment import EmailEnvironment
    from models import EmailAction, ActionType, Priority
    import random
    env = EmailEnvironment(task_id=2)
    env.reset()
    rewards = []
    for _ in range(5):
        if env._done:
            break
        a = EmailAction(action_type=random.choice(list(ActionType)), classification=random.choice(list(Priority)))
        r = env.step(a)
        rewards.append(r.reward.total)
        if r.done:
            break
    assert len(rewards) >= 3, "Need ≥3 steps to verify dense rewards"
    non_zero = sum(1 for r in rewards if r != 0.0)
    assert non_zero >= 2, f"Reward appears sparse — only {non_zero}/{len(rewards)} non-zero"
    return f"Dense: {len(rewards)} steps, rewards={[round(r,3) for r in rewards]}"

def check_urgent_penalty():
    from server.environment import EmailEnvironment
    from models import EmailAction, ActionType, Priority
    # Find an urgent email and archive it — should get large negative reward
    env = EmailEnvironment(task_id=2)
    env.reset()
    urgent_reward = None
    while not env._done:
        email = env._emails[env._index]
        if email["label"] == "urgent":
            a = EmailAction(action_type=ActionType.archive, classification=Priority.spam)
            r = env.step(a)
            urgent_reward = r.reward.total
            break
        else:
            a = EmailAction(action_type=ActionType.archive, classification=Priority.spam)
            env.step(a)
    assert urgent_reward is not None, "No urgent emails found"
    assert urgent_reward < -1.0, f"Missing urgent penalty: reward={urgent_reward} should be < -1.0"
    return f"Urgent missed penalty: {urgent_reward}"


# ── 6. DATASET ───────────────────────────────────────────────────────────────
def check_dataset_section():
    print("\n📊 Dataset")

def check_emails_json():
    with open(ROOT / "data" / "emails.json") as f:
        emails = json.load(f)
    assert len(emails) >= 10, f"Need ≥10 emails, got {len(emails)}"
    required_fields = {"id", "subject", "body", "sender", "label", "expected_action"}
    for e in emails:
        missing = required_fields - set(e.keys())
        assert not missing, f"Email {e.get('id')} missing fields: {missing}"
    labels = {e["label"] for e in emails}
    assert "urgent" in labels and "spam" in labels, f"Need urgent+spam labels, got: {labels}"
    return f"{len(emails)} emails, labels: {sorted(labels)}"

def check_env_vars_documented():
    with open(ROOT / "inference.py") as f:
        src = f.read()
    for var in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
        assert var in src, f"inference.py must reference env var {var}"
    return "API_BASE_URL, MODEL_NAME, HF_TOKEN all referenced"

def check_openai_client_used():
    with open(ROOT / "inference.py") as f:
        src = f.read()
    assert "openai" in src.lower(), "inference.py must use OpenAI client"
    assert "OpenAI(" in src, "Must instantiate OpenAI client"
    return "OpenAI client instantiated"


# ── 7. DOCKER ────────────────────────────────────────────────────────────────
def check_docker_section():
    print("\n🐳 Docker")

def check_dockerfile_exists():
    df = ROOT / "Dockerfile"
    assert df.exists()
    content = df.read_text()
    assert "FROM python" in content, "Dockerfile must use a Python base image"
    assert "EXPOSE 8000" in content, "Dockerfile must EXPOSE 8000"
    assert "uvicorn" in content, "Dockerfile must run uvicorn"
    return "Dockerfile valid — FROM python, EXPOSE 8000, uvicorn CMD"

def check_requirements_txt():
    req = ROOT / "server" / "requirements.txt"
    assert req.exists()
    content = req.read_text()
    for pkg in ["fastapi", "uvicorn", "pydantic"]:
        assert pkg in content, f"requirements.txt must include {pkg}"
    return "requirements.txt includes fastapi, uvicorn, pydantic"


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Inbox OS — pre-submission validator")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  AUTONOMOUS INBOX OS — Pre-Submission Validator")
    print("=" * 60)

    # Check for yaml
    try:
        import yaml
    except ImportError:
        print("  ⚠️  PyYAML not installed — skipping yaml checks (pip install pyyaml)")
        yaml = None

    checks = [
        # Files
        (check_files, True),
        (check_inference_at_root, True),
        (check_openenv_yaml if yaml else lambda: (_ for _ in ()).throw(Exception("PyYAML not installed")), yaml is not None),
        # Models
        (check_models, False),  # section header
        (check_models_importable, True),
        (check_action_model, True),
        (check_observation_model, True),
        # Environment
        (check_environment, False),
        (check_env_importable, True),
        (check_reset, True),
        (check_step, True),
        (check_state, True),
        (check_episode_completes, True),
        # Tasks
        (check_tasks_section, False),
        (check_three_tasks, True),
        (check_grader_scores, True),
        (check_grader_deterministic, True),
        # Reward
        (check_reward_section, False),
        (check_reward_is_dense, True),
        (check_urgent_penalty, True),
        # Dataset
        (check_dataset_section, False),
        (check_emails_json, True),
        (check_env_vars_documented, True),
        (check_openai_client_used, True),
        # Docker
        (check_docker_section, False),
        (check_dockerfile_exists, True),
        (check_requirements_txt, True),
    ]

    section_fns = {check_files, check_models, check_environment,
                   check_tasks_section, check_reward_section,
                   check_dataset_section, check_docker_section}

    passed = 0
    failed = 0
    critical_fail = False

    for fn, critical in checks:
        if fn in section_fns:
            fn()  # section headers just print
            continue
        ok = check(fn.__name__.replace("check_", "").replace("_", " "), fn, critical)
        if ok:
            passed += 1
        else:
            failed += 1
            if critical:
                critical_fail = True

    print("\n" + "=" * 60)
    print(f"  Passed: {passed}  |  Failed: {failed}")

    if critical_fail:
        print("  ❌ SUBMISSION BLOCKED — fix critical failures before submitting")
        sys.exit(1)
    elif failed > 0:
        print("  ⚠️  Warnings present — review before submitting")
        sys.exit(0)
    else:
        print("  ✅ ALL CHECKS PASSED — ready to submit!")
        sys.exit(0)
