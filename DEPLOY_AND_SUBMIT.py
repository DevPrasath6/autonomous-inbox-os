#!/usr/bin/env python3
"""
DEPLOY_AND_SUBMIT.md — Step-by-step deployment guide

Run this file to print the guide:
    python DEPLOY_AND_SUBMIT.py
"""

GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║       AUTONOMOUS INBOX OS — DEPLOYMENT & SUBMISSION GUIDE       ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Test locally with Docker
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    docker build -t autonomous-inbox-os .
    docker run -p 8000:8000 autonomous-inbox-os

Verify it works:
    curl http://localhost:8000/           # should return {"status": "running", ...}
    curl -X POST http://localhost:8000/reset?task_id=2
    open http://localhost:8000/demo       # interactive UI

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Run the pre-submission validator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python validate.py

All checks must show ✅ before you submit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Run the baseline inference script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=your_hf_token_here

    python inference.py

This produces baseline_scores.json — copy the scores into README.md.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — Prepare Hugging Face Space README
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The HF Space README must start with the YAML front-matter header.
Prepend HF_README_HEADER.md to your README.md:

    cat HF_README_HEADER.md README.md > README_HF.md
    # Then use README_HF.md as the README in your HF Space repo

The front-matter tells HF Spaces:
    - sdk: docker        (use our Dockerfile)
    - tags: openenv      (required for hackathon discovery)
    - emoji/colors       (visual identity in the Space catalog)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — Deploy to Hugging Face Spaces
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option A — OpenEnv CLI (recommended):
    pip install openenv
    openenv push --repo-id your-username/autonomous-inbox-os

Option B — Manual git push:
    # Create a new Space at https://huggingface.co/new-space
    # Choose: SDK = Docker, Visibility = Public
    git clone https://huggingface.co/spaces/YOUR_USERNAME/autonomous-inbox-os
    cp -r . autonomous-inbox-os/
    cd autonomous-inbox-os
    cat HF_README_HEADER.md README.md > README.md  # prepend HF header
    git add .
    git commit -m "Initial deployment"
    git push

After pushing, HF will build your Docker image automatically.
Wait for the build to complete (2-5 minutes), then verify:
    curl https://YOUR_USERNAME-autonomous-inbox-os.hf.space/
    curl -X POST https://YOUR_USERNAME-autonomous-inbox-os.hf.space/reset

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — Pre-submission checklist (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before clicking submit, verify EVERY item:

  [ ] HF Space deploys → GET / returns 200 with {"status": "running"}
  [ ] POST /reset returns a valid EmailObservation JSON
  [ ] POST /step accepts an EmailAction and returns StepResult
  [ ] GET /grader returns {"score": float, "details": {...}}
  [ ] openenv validate passes (or equivalent spec check)
  [ ] docker build && docker run works cleanly
  [ ] inference.py is at the ROOT of the project (not in a subfolder)
  [ ] inference.py runs end-to-end without errors
  [ ] baseline_scores.json was produced and scores are in README.md
  [ ] 3+ tasks defined with graders that return scores in [0.0, 1.0]
  [ ] README.md has: description, action/obs spaces, tasks, setup, scores
  [ ] openenv.yaml is present with all required fields
  [ ] Environment variables API_BASE_URL, MODEL_NAME, HF_TOKEN are used

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENT VARIABLES (set before running inference.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    API_BASE_URL   The API endpoint, e.g. https://api.openai.com/v1
    MODEL_NAME     The model ID, e.g. gpt-4o-mini or meta-llama/Llama-3-8b
    HF_TOKEN       Your Hugging Face API key / OpenAI API key

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING BREAKDOWN (know what judges look for)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Real-world utility    30%  ← Stress engine + multi-action = full marks
    Task & grader quality 25%  ← 4 tasks, deterministic, 0-1 range
    Environment design    20%  ← Dense reward, clean reset, sensible episodes
    Code quality          15%  ← Typed Pydantic models, clean structure
    Creativity & novelty  10%  ← Stress engine is novel in OpenEnv catalog

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEMO SCRIPT (for live presentation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Open /demo in browser
2. Select Task 3 (Hard)
3. Click "Flood Inbox" — show stress spike to 30%+
4. Click "Step AI" 3-4 times — show reward flashes
5. Click "Auto Run" — let it run to completion
6. Point out: stress dropping as AI handles emails correctly
7. Show the final score and metrics panel
8. Say: "This isn't a classifier. It's an AI that manages responsibility."

"""

if __name__ == "__main__":
    print(GUIDE)
