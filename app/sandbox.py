#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sandbox.py — thin wrapper for executing llm.py safely.

- Provides run_sandbox(cmd, env, cwd) used by agent.py.
- Captures stdout/stderr.
- Returns dict with parsed JSON or error.
"""

import subprocess, json

def run_sandbox(cmd, env=None, cwd=None):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            env=env,
            cwd=cwd
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
