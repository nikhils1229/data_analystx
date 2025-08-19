import os
import re
import json
import subprocess
import traceback
from fastapi import FastAPI, Request
from openai import OpenAI

app = FastAPI()
client = OpenAI()

# ----------------------------
# Preprocessing: Normalize Questions
# ----------------------------

def normalize_questions(questions_txt: str) -> dict:
    """Cast messy input into a JSON dict {question: ""}."""
    # Case 1: Already valid JSON
    try:
        q_obj = json.loads(questions_txt)
        if isinstance(q_obj, dict) and q_obj:
            return q_obj
    except Exception:
        pass

    # Case 2: Try LLM to cast into JSON
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a preprocessing assistant for a Data Analyst Agent. "
                        "Convert the following input into a clean JSON object where "
                        "each question is a key and the value is an empty string. "
                        "If the input is descriptive text, extract the questions. "
                        "Output only JSON, no explanations, no markdown fences."
                    ),
                },
                {"role": "user", "content": questions_txt},
            ],
            max_tokens=1500,
        )
        cleaned = resp.choices[0].message.content.strip()

        # 🔧 Remove code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9]*", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

        q_obj = json.loads(cleaned)
        if isinstance(q_obj, dict) and q_obj:
            return q_obj
    except Exception as e:
        print("DEBUG normalize_questions error:", e)

    # Case 3: Fallback → split by lines
    lines = [l.strip() for l in questions_txt.splitlines() if l.strip()]
    if not lines:
        return {"What columns are available?": ""}
    return {line: "" for line in lines}

# ----------------------------
# Run llm.py with QUESTIONS_JSON
# ----------------------------

def run_llm(questions: dict) -> dict:
    """Run llm.py inside sandbox with QUESTIONS_JSON env var."""
    try:
        env = os.environ.copy()
        env["QUESTIONS_JSON"] = json.dumps(questions)
        result = subprocess.run(
            ["python3", "llm.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=120,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            print("DEBUG llm.py stderr:", result.stderr)
            return {"error": result.stderr}
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# ----------------------------
# FastAPI endpoint
# ----------------------------

@app.post("/api/")
async def api(request: Request):
    try:
        payload = await request.json()
        questions_txt = payload.get("input", "")
        print("DEBUG raw input:", questions_txt)

        questions = normalize_questions(questions_txt)
        print("DEBUG normalized questions:", questions)

        answers = run_llm(questions)
        print("DEBUG answers:", answers)

        # promptfoo expects a stringified JSON list under 'output'
        return {"output": json.dumps([answers])}
    except Exception as e:
        print("DEBUG api error:", traceback.format_exc())
        return {"output": json.dumps([{"error": str(e)}])}
