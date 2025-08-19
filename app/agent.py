#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent.py — Orchestrator for the data analyst agent.

- Accepts uploaded questions file / JSON.
- Calls sandboxed execution (llm.py).
- Ensures only valid JSON is returned to the API.
- Handles errors gracefully (never raw stack traces).
"""

import os
import json
import subprocess
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI()

# ----------------------------
# Sandbox runner
# ----------------------------

def run_sandbox(cmd):
    """
    Run the sandboxed process (llm.py) and capture stdout/stderr.
    Always return a dict.
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        raw_out = (proc.stdout or "").strip()
        raw_err = (proc.stderr or "").strip()

        if not raw_out:
            return {"error": f"Sandbox produced no output", "stderr": raw_err}

        try:
            return json.loads(raw_out)
        except Exception as e:
            return {
                "error": f"Invalid JSON from sandbox: {e}",
                "raw": raw_out,
                "stderr": raw_err,
            }

    except Exception as e:
        return {"error": f"Sandbox failed to run: {e}"}

# ----------------------------
# API endpoint
# ----------------------------

@app.post("/api/")
async def analyze(
    questions_txt: Optional[UploadFile] = File(None),
    questions_json: Optional[str] = Form(None)
):
    """
    Main API endpoint.
    Accepts questions either as uploaded text file or as JSON string.
    Returns sandbox answers as JSON.
    """

    q_path = None
    q_env = None

    # If questions.txt provided
    if questions_txt:
        q_path = "questions.json"
        content = (await questions_txt.read()).decode("utf-8").strip()
        try:
            # try parse directly as JSON
            q_obj = json.loads(content)
        except Exception:
            # fallback: simple heuristic (one question per line)
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            q_obj = lines
        with open(q_path, "w", encoding="utf-8") as f:
            json.dump(q_obj, f, ensure_ascii=False)

    # If questions_json provided via form
    elif questions_json:
        q_env = questions_json

    # Build environment
    env = os.environ.copy()
    if q_path:
        env["QUESTIONS_PATH"] = q_path
    if q_env:
        env["QUESTIONS_JSON"] = q_env

    # Call sandbox (llm.py)
    result = run_sandbox(["python3", "llm.py"])

    # Return JSON response (400 if error)
    if "error" in result:
        return JSONResponse(content=result, status_code=400)
    return JSONResponse(content=result, status_code=200)
