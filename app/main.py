#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — FastAPI entrypoint for the Data Analyst Agent API.

- Accepts `questions.txt` and optional data files via POST /api/.
- Reads content and passes it to agent.py runner.
- Returns JSON result or clean error.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Optional
import json
from . import agent

app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses LLMs to source, prepare, analyze, and visualize data.",
)


@app.post("/api/")
async def analyze_data(
    questions_txt: UploadFile = File(..., description="A .txt file containing the data analysis questions."),
    files: Optional[List[UploadFile]] = File(
        None, description="Optional data files (e.g., .csv, .json, .parquet, .tar.gz)."
    ),
):
    """
    Accept a data analysis task and return the result.
    """

    if not questions_txt.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="A 'questions.txt' file is required.")

    # Read the questions.txt content
    question_content_bytes = await questions_txt.read()
    question_content_str = question_content_bytes.decode("utf-8")

    # Prepare dict of attached files (filename → bytes)
    attached_files = {}
    if files:
        for f in files:
            attached_files[f.filename] = await f.read()

    # Ensure questions.txt is also in attached files
    attached_files[questions_txt.filename] = question_content_bytes

    # Run the agent
    try:
        result = await agent.run_data_analyst_agent(question_content_str, attached_files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result)

    return result


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Data Analyst Agent API. Please POST to /api/ to submit a task."
    }
