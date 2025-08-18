import openai
from . import config
import json

# Configure the OpenAI client
# The client automatically uses the OPENAI_API_KEY environment variable if not passed directly.
client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

def get_plan_from_llm(user_request: str, filenames: list[str]) -> list[str]:
    """
    Sends the user's request to the OpenAI API to get a step-by-step plan.
    """
    # The prompt is fundamentally the same, just placed inside the API's message structure.
    system_prompt = "You are a master data analyst agent. Your task is to break down a user's request into a series of clear, executable steps. Do not answer the questions directly or generate code. Respond with a JSON array of strings, where each string is a step in the plan."
    user_prompt = f"""
    User Request:
    ---
    {user_request}
    ---
    Available Files: {', '.join(filenames) if filenames else 'None'}
    
    Example Response:
    ["Scrape the data from the provided URL.", "Load the scraped data into a pandas DataFrame named 'df'.", "Calculate the answer for the first question.", "Generate a scatterplot as requested.", "Assemble the final answers into a JSON array."]

    Provide only the JSON array of steps.
    """
    
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"} # Use JSON mode for reliable output
        )
        plan_str = response.choices[0].message.content
        # OpenAI's JSON mode should return a valid JSON object, but we parse it just in case
        # It might return a structure like {"plan": [...]}, so we need to be flexible.
        plan_data = json.loads(plan_str)
        
        # Check if the data is a dict with a key (e.g., "plan") or just a list
        if isinstance(plan_data, dict):
            # Find the list within the dict
            for key, value in plan_data.items():
                if isinstance(value, list):
                    plan = value
                    break
            else:
                 raise ValueError("JSON object from AI does not contain a list of steps.")
        elif isinstance(plan_data, list):
            plan = plan_data
        else:
            raise ValueError("Unexpected JSON format from AI.")

        print(f"✅ AI Plan Generated: {plan}")
        return plan
    except (Exception) as e:
        print(f"❌ Error generating plan with OpenAI: {e}")
        return ["Error: Could not generate a plan."]

def get_code_from_llm(task: str, context: str) -> str:
    """
    Sends a specific task and context to the OpenAI API to get Python code.
    """
    system_prompt = """
    You are a Python code generation expert. Given the context of previous steps and the current task, write the Python code to complete ONLY the current task.
    - Only write Python code. Do not add any explanations, comments, or markdown formatting like ```python.
    - If the task is to generate a plot, save it to an in-memory buffer (`io.BytesIO`) and encode it into a base64 data URI string. The final output must be a single `print()` statement containing ONLY the data URI.
    - If the task is a calculation or data manipulation, print the result to standard output so it can be captured.
    - If the task is to assemble the final answer, print the final JSON structure.
    - Assume all necessary libraries (pandas, matplotlib, etc.) are already installed.
    - Make use of any provided data files by reading them (e.g., pd.read_csv('filename.csv')).
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
    
    Respond with only the raw Python code for this task.
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
        # Clean up potential markdown formatting if the model still adds it
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
            
        print(f"✅ AI Code Generated for task '{task}'")
        return code.strip()
    except (Exception) as e:
        print(f"❌ Error generating code with OpenAI: {e}")
        return f"print('Error generating code: {e}')"
