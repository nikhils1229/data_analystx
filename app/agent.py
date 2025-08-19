#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent.py — Orchestrator for the Data Analyst Agent.

- Provides `run_data_analyst_agent` for use by main.py.
- Handles saving questions + files to sandbox.
- Runs llm.py safely (stdout only JSON, stderr captured).
- Always returns dict or list, never raw text.
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, Any


def run_sandbox(cmd, env=None, cwd=None):
    """
    Run sandbox process (llm.py) and capture stdout/stderr.
    Always return dict (possibly with {"error": ...}).
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            env=env,
            cwd=cwd,
        )
        raw_out = (proc.stdout or "").strip()
        raw_err = (proc.stderr or "").strip()

        if not raw_out:
            return {"error": "Sandbox produced no output", "stderr": raw_err}

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


async def run_data_analyst_agent(
    questions_txt: str, files: Dict[str, bytes]
) -> Any:
    """
    Main entrypoint used by main.py.
    - questions_txt: content of questions.txt as string
    - files: dict of {filename: file_bytes}
    Returns: dict or list (answers) or {"error": "..."}
    """
    try:
        # Create temp working directory for sandbox
        with tempfile.TemporaryDirectory() as tmpdir:
            q_path = os.path.join(tmpdir, "questions.json")

            # Try parse as JSON
            try:
                q_obj = json.loads(questions_txt)
            except Exception:
                # fallback: treat as text → one question per line
                lines = [
                    l.strip() for l in questions_txt.splitlines() if l.strip()
                ]
                q_obj = lines

            with open(q_path, "w", encoding="utf-8") as f:
                json.dump(q_obj, f, ensure_ascii=False)

            # Save uploaded files into tmpdir
            for fname, content in files.items():
                fpath = os.path.join(tmpdir, fname)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "wb") as f:
                    f.write(content)

            # Prepare environment
            env = os.environ.copy()
            env["QUESTIONS_PATH"] = q_path

            # 🔧 Absolute path to llm.py inside repo
            llm_path = os.path.join(os.path.dirname(__file__), "llm.py")

            # Run llm.py inside sandbox
            result = run_sandbox(["python3", llm_path], env=env, cwd=tmpdir)

            return result
    except Exception as e:
        return {"error": f"Agent orchestration failed: {e}"}
