# 📬 Autonomous Inbox OS

> **An AI agent simulation environment where an agent manages a flooded executive inbox under real-world constraints: time pressure, cognitive load, conflicting priorities, and multi-step decision-making.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/🤗-HF%20Spaces-yellow)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-ready-0db7ed)](https://hub.docker.com)

---

## 🧠 Problem

Every day, professionals spend hours managing emails — not because it's hard, but because it's **overwhelming**. Important messages get buried. Deadlines are missed. Stress builds up.

Traditional classifiers answer: *"Is this spam?"*  
This environment asks: *"What would a skilled executive assistant DO with this?"*

---

## 🚀 What This Environment Simulates

**Demo Link -**  https://devv16-autonomous-inbox-os.hf.space/demo

**Autonomous Inbox OS** is a real-world decision-making environment where an AI agent must:

- **Classify** emails by priority (urgent / important / low_priority / spam)
- **Take actions**: reply, escalate, archive, schedule, or ignore
- **Manage cognitive load** — a stress engine models inbox overload
- **Work under time pressure** — a limited action budget per episode
- **Handle conflicting priorities** — double-booked meetings, legal deadlines, security breaches

The key innovation: **we model human cognitive load as a first-class reward signal**. The agent is not just rewarded for correctness — it's penalized for letting stress spiral and missing urgent items.

---

## 🏗️ Architecture

```
autonomous_inbox_os/
├── models.py              # Pydantic models (Action, Observation, State, Reward)
├── inference.py           # Baseline inference script (root, uses OpenAI client)
├── openenv.yaml           # OpenEnv manifest
├── Dockerfile             # Container definition
├── README.md
├── data/
│   └── emails.json        # 15 realistic emails across 8 categories
├── server/
│   ├── environment.py     # Core OpenEnv logic (step/reset/state)
│   ├── app.py             # FastAPI server
│   └── requirements.txt
└── static/
    └── index.html         # Interactive demo frontend
```

---

## 📐 Observation Space

| Field | Type | Description |
|---|---|---|
| `email_id` | int | Unique email ID |
| `subject` | str | Email subject line |
| `body` | str | Email body text |
| `sender` | str | Sender address |
| `has_attachment` | bool | Attachment present |
| `meeting_request` | bool | Meeting invitation detected |
| `inbox_size` | int | Remaining emails in episode |
| `pending_urgent` | int | Unprocessed urgent emails |
| `pending_important` | int | Unprocessed important emails |
| `user_stress_level` | float [0,1] | Current cognitive load (stress engine) |
| `time_remaining` | int | Steps left before episode timeout |
| `step_count` | int | Current step index |

---

## 🎮 Action Space

```json
{
  "action_type": "escalate | reply | archive | schedule | classify | ignore",
  "classification": "urgent | important | low_priority | spam",
  "reply_text": "optional reply body",
  "escalate_to": "optional escalation target",
  "scheduled_time": "ISO datetime string",
  "reasoning": "chain-of-thought explanation"
}
```

---

## 🏆 Tasks

### Task 1 — Spam Detection `[Easy]`
Detect and archive spam emails. Mix of spam, newsletters, transactional, and HR emails.  
**Target score:** ≥ 0.80

### Task 2 — Inbox Prioritization `[Medium]`
Classify all 15 emails across all categories and take the correct action for each.  
**Target score:** ≥ 0.60

### Task 3 — Executive Decision-Making Under Pressure `[Hard]`
Handle a flooded inbox (doubled urgent emails), conflicting priorities, legal deadlines, data breaches. Agent must manage stress while maintaining accuracy.  
**Target score:** ≥ 0.45

---

## 🧮 Reward Function

Dense reward signal — provided at **every step**, not just episode end:

```python
reward = (
    classification_accuracy * 0.35    # Correct label prediction
  + action_correctness * 0.35         # Correct action taken
  + reply_quality * 0.20              # Quality of generated reply
  - 0.05                              # Time cost (every step)
  - stress_level * 0.20               # Cognitive load penalty
  - 2.00 * missed_urgent              # Missed urgent email penalty
  + 0.10 * has_reasoning              # Chain-of-thought bonus
)
```

**Range:** approximately [-3.0, +2.0] per step

---

## 🔧 Setup & Usage

### Docker (recommended)

```bash
docker build -t autonomous-inbox-os .
docker run -p 8000:8000 autonomous-inbox-os
```

### Local

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset?task_id=2` | POST | Start a new episode |
| `/step` | POST | Submit an action, get observation+reward |
| `/state` | GET | Current episode state |
| `/tasks` | GET | List all tasks |
| `/grader` | GET | Final episode score (0.0–1.0) |
| `/metrics` | GET | Live performance metrics |
| `/simulate?task_id=2&policy=random` | POST | Run full episode with built-in policy |

---

## 🤖 Baseline Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token_here

python inference.py
```

### Baseline Scores (GPT-4o-mini)

| Task | Difficulty | Score |
|---|---|---|
| Task 1 (Spam Detection) | Easy | ~0.81 |
| Task 2 (Prioritization) | Medium | ~0.63 |
| Task 3 (Under Pressure) | Hard | ~0.44 |
| **Average** | | **~0.63** |

---

## 🎥 Demo

Open `static/index.html` in your browser (or served via the Docker container at `/`):

- **Start Episode** — Load inbox and begin
- **Step AI** — Let the agent process one email
- **Auto Run** — Watch the agent run the full inbox
- **Flood Inbox** — Simulate inbox overload and watch stress spike

---

## 🔥 What Makes This Different

1. **Stress Engine** — Models human cognitive load as a quantified, dynamic variable. The agent is penalized for letting stress grow, rewarded for managing it down.
2. **Multi-action decisions** — Not just classification. Escalate, reply, schedule, archive — each with its own reward signal.
3. **Time pressure** — Limited action budget creates urgency. Hard task adds double urgents to simulate realistic flood scenarios.
4. **Chain-of-thought bonus** — Agents that provide reasoning get a small reward bonus, incentivizing explainable AI behavior.
5. **Dense reward shaping** — Every step provides a signal. No sparse end-of-episode reward.

---

## 📄 License

MIT
