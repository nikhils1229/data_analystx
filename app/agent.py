import os
import json
import tempfile
import subprocess
import traceback
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()

def normalize_questions(questions_txt: str) -> dict:
    """Try to cast user input into JSON dict of questions → ''"""
    try:
        return json.loads(questions_txt)
    except Exception:
        pass

    # Use LLM to cast into JSON dict
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
                        "If the input is descriptive text, extract the questions "
                        "and output only JSON, nothing else."
                    ),
                },
                {"role": "user", "content": questions_txt},
            ],
            max_tokens=1500,
        )
        cleaned = resp.choices[0].message.content.strip()
        return json.loads(cleaned)
    except Exception as e:
        # Fallback: each line is a question
        lines = [l.strip() for l in questions_txt.splitlines() if l.strip()]
        return {line: "" for line in lines}

def run_llm(questions: dict) -> dict:
    """Run llm.py inside sandbox with QUESTIONS_JSON env var"""
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
            return {"error": result.stderr}
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.route("/api/", methods=["POST"])
def api():
    try:
        payload = request.get_json(force=True)
        questions_txt = payload.get("input", "")
        questions = normalize_questions(questions_txt)
        answers = run_llm(questions)
        # promptfoo expects a stringified JSON list under 'output'
        return jsonify({"output": json.dumps([answers])})
    except Exception as e:
        return jsonify({"output": json.dumps([{"error": str(e)}])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
