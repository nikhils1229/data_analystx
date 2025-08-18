import openai
from . import config
import json

# Configure the OpenAI client
client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

def get_plan_from_llm(user_request: str, filenames: list[str]) -> list[str]:
    system_prompt = (
        "You are a master data analyst agent. Your task is to break down a user's request "
        "into a series of clear, executable steps. Do not answer the questions directly or generate code. "
        "Respond with a JSON array of strings, where each string is a step in the plan."
    )
    user_prompt = f"""
    User Request:
    ---
    {user_request}
    ---
    Available Files (local only, no internet in execution): {', '.join(filenames) if filenames else 'None'}
    
    Constraints:
    - The execution environment CANNOT access the internet.
    - All data must be read from the provided local files.
    - If the request references a URL, assume the HTML has been provided locally.

    Example Response:
    ["Load the provided local HTML file(s) with pandas.read_html.", 
     "Extract the 'Highest-grossing films' table into a DataFrame.", 
     "Compute answers (counts, earliest film over thresholds, correlations).", 
     "Generate the plot and return a base64 data URI under 100000 bytes.",
     "Assemble the final JSON array of strings and print it."]
    """

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        plan_str = response.choices[0].message.content
        plan_data = json.loads(plan_str)

        if isinstance(plan_data, dict):
            for key, value in plan_data.items():
                if isinstance(value, list):
                    return value
            raise ValueError("JSON object from AI does not contain a list of steps.")
        elif isinstance(plan_data, list):
            return plan_data
        else:
            raise ValueError("Unexpected JSON format from AI.")

    except Exception as e:
        print(f"❌ Error generating plan with OpenAI: {e}")
        return ["Error: Could not generate a plan."]

def get_code_from_llm(task: str, context: str) -> str:
    """
    Generate Python code for a specific task.
    """
    system_prompt = """
You are a Python code generation agent.
CRITICAL RULES:
- Output ONLY raw Python code (no prose, no markdown).
- Never refuse tasks. Always generate working Python code.
- Do NOT attempt to fetch from the internet (requests, urllib, etc. are forbidden).
- Assume any web page mentioned has already been saved locally as .html.
- To parse tables from HTML: use pandas.read_html('localfile.html').
- For calculations: print() the result directly.
- For final answers: print(json.dumps([...])) with a JSON array of strings.
- For plots:
  * Use matplotlib.
  * Save figure to BytesIO as PNG.
  * Base64 encode it.
  * Print only the data URI (string starting with data:image/png;base64,).
  * Keep file < 100000 bytes by reducing figure size/DPI if needed.
"""
    user_prompt = f"""
Context:
---
{context if context else 'This is the first step.'}
---

Task:
---
{task}
---
"""

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        code = response.choices[0].message.content.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()
    except Exception as e:
        print(f"❌ Error generating code with OpenAI: {e}")
        return f"print('Error generating code: {e}')"

def answer_questions_directly(user_request: str) -> list[str]:
    system_prompt = (
        "You are a helpful assistant. Answer the user's questions directly. "
        "Return ONLY a JSON array of strings (each string is an answer)."
    )
    user_prompt = f"Questions:\n{user_request}"

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            if all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
                return list(parsed.values())
            raise ValueError(f"Unexpected dict format: {parsed}")
        raise ValueError(f"Unexpected response format: {parsed}")

    except Exception as e:
        print(f"❌ Error in direct answering: {e}")
        return [f"Error answering questions: {e}"]
