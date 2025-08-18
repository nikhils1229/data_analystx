import openai
from . import config
import json

# Configure the OpenAI client
client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

def get_plan_from_llm(user_request: str, filenames: list[str]) -> list[str]:
    """
    Sends the user's request to the OpenAI API to get a step-by-step plan.
    """
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
    Available Files: {', '.join(filenames) if filenames else 'None'}
    
    Example Response:
    ["Load the dataset from 'data.csv'.", "Compute correlation between Rank and Peak.", "Assemble the final JSON answers."]
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
    Sends a specific task and context to the OpenAI API to get Python code.
    """
    system_prompt = """
    You are a Python code generation expert. Given the context of previous steps and the current task, 
    write the Python code to complete ONLY the current task.
    - Only write Python code. Do not add explanations or markdown formatting.
    - If the task is to generate a plot, save it to an in-memory buffer (`io.BytesIO`) and encode it into a base64 data URI string. Print only the data URI.
    - If the task is a calculation, print the result.
    - If the task is to assemble the final answer, print the final JSON structure.
    - Assume all necessary libraries (pandas, matplotlib, etc.) are installed.
    """
    user_prompt = f"""
    Context from previous steps:
    ---
    {context if context else 'This is the first step.'}
    ---

    Current Task:
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
    """
    When no files are provided, answer the questions directly with the LLM
    and return a JSON array of strings.
    Accepts:
    - a raw list of answers
    - a dict with a list inside
    - a dict of key:value pairs (returns just the values)
    """
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

        # Case 1: direct list
        if isinstance(parsed, list):
            return parsed

        # Case 2: dict with a list inside
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            # Case 3: dict of string:string → take the values
            if all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
                return list(parsed.values())

            raise ValueError(f"Unexpected dict format: {parsed}")

        raise ValueError(f"Unexpected response format: {parsed}")

    except Exception as e:
        print(f"❌ Error in direct answering: {e}")
        return [f"Error answering questions: {e}"]
