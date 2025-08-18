from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import List, Optional
from . import agent
from . import sandbox
import json

app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses LLMs to source, prepare, analyze, and visualize data.",
)

@app.on_event("startup")
async def startup_event():
    """Build the Docker image on application startup."""
    sandbox.build_docker_image()


@app.post("/api/")
async def analyze_data(
    questions_txt: UploadFile = File(..., description="A .txt file containing the data analysis questions."),
    files: Optional[List[UploadFile]] = File(None, description="Optional data files (e.g., .csv, .png).")
):
    """
    This endpoint accepts a data analysis task and returns the result.
    """
    if not questions_txt.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="A 'questions.txt' file is always required.")

    # Read the content of the main questions file
    question_content_bytes = await questions_txt.read()
    question_content_str = question_content_bytes.decode('utf-8')

    # Prepare attached files to be passed to the agent
    attached_files = {}
    if files:
        for file in files:
            attached_files[file.filename] = await file.read()
    
    # Add the questions.txt to the files dict so the sandbox can read it too
    attached_files[questions_txt.filename] = question_content_bytes

    # Trigger the agent workflow
    result = await agent.run_data_analyst_agent(question_content_str, attached_files)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result)
        
    return result

@app.get("/")
def read_root():
    return {"message": "Welcome to the Data Analyst Agent API. Please POST to /api/ to submit a task."}
