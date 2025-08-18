from . import llm
from . import sandbox
import json

MAX_REFINEMENT_ATTEMPTS = 2

async def run_data_analyst_agent(question_content_str: str, files: dict[str, bytes]) -> dict:
    """
    The main orchestrator for the data analyst agent.
    """
    # 1. Get a plan from the LLM
    filenames = list(files.keys())
    plan = llm.get_plan_from_llm(question_content_str, filenames)

    if "Error" in plan[0]:
        return {"error": "Could not generate a plan for the request."}

    context = f"User request: {question_content_str}\nAvailable files: {', '.join(filenames)}"
    final_result = ""

    # 2. Execute the plan step-by-step
    for task in plan:
        print(f"🚀 Executing task: {task}")
        
        # Initial code generation
        code_to_run = llm.get_code_from_llm(task, context)
        
        for attempt in range(MAX_REFINEMENT_ATTEMPTS):
            # 3. Run code in the sandbox
            stdout, stderr = sandbox.run_code_in_sandbox(code_to_run, files)

            if stderr:
                print(f"⚠️ Error detected on attempt {attempt + 1}. Refining code...")
                # 4. Refine code on error
                refinement_context = f"{context}\n\nThe previous code attempt failed. Please fix it.\nCode:\n{code_to_run}\n\nError:\n{stderr}"
                code_to_run = llm.get_code_from_llm(task, refinement_context)
                if attempt == MAX_REFINEMENT_ATTEMPTS - 1:
                     print("❌ Max refinement attempts reached. Failing task.")
                     return {"error": f"Task failed after multiple attempts: {task}", "details": stderr}
            else:
                # Success! Update context and move to the next task
                context += f"\n\nResult of '{task}':\n{stdout}"
                final_result = stdout # Store the latest output as the potential final result
                break # Exit refinement loop
    
    # 5. Format and return the final result
    try:
        # The last step should have printed the final JSON
        return json.loads(final_result)
    except json.JSONDecodeError:
        # If the final output isn't valid JSON, return it as a string in an object
        return {"result": final_result}
