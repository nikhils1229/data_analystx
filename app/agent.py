from . import llm
from . import sandbox
import json

MAX_REFINEMENT_ATTEMPTS = 2

async def run_data_analyst_agent(question_content_str: str, files: dict[str, bytes]) -> dict:
    """
    The main orchestrator for the data analyst agent.
    If no files are provided (besides questions.txt), falls back to direct LLM answering.
    """
    # Exclude questions.txt from the dataset list
    filenames = [f for f in files.keys() if not f.endswith("questions.txt")]

    # 🔹 If no data files, fall back to direct LLM Q&A
    if not filenames:
        print("⚡ No data files detected. Falling back to direct LLM answering.")
        try:
            answers = llm.answer_questions_directly(question_content_str)
            return {"answers": answers}
        except Exception as e:
            return {"error": f"Direct answering failed: {e}"}

    # 🔹 Otherwise, follow the data-analysis pipeline
    plan = llm.get_plan_from_llm(question_content_str, filenames)
    if "Error" in plan[0]:
        return {"error": "Could not generate a plan for the request."}

    context = f"User request: {question_content_str}\nAvailable files: {', '.join(filenames)}"
    final_result = ""

    for task in plan:
        print(f"🚀 Executing task: {task}")
        code_to_run = llm.get_code_from_llm(task, context)

        for attempt in range(MAX_REFINEMENT_ATTEMPTS):
            stdout, stderr = sandbox.run_code_in_sandbox(code_to_run, files)

            if stderr:
                print(f"⚠️ Error detected on attempt {attempt + 1}. Refining code...")
                refinement_context = (
                    f"{context}\n\nThe previous code attempt failed. Please fix it.\n"
                    f"Code:\n{code_to_run}\n\nError:\n{stderr}"
                )
                code_to_run = llm.get_code_from_llm(task, refinement_context)
                if attempt == MAX_REFINEMENT_ATTEMPTS - 1:
                    print("❌ Max refinement attempts reached. Failing task.")
                    return {"error": f"Task failed after multiple attempts: {task}", "details": stderr}
            else:
                context += f"\n\nResult of '{task}':\n{stdout}"
                final_result = stdout
                break

    try:
        return json.loads(final_result)
    except json.JSONDecodeError:
        return {"result": final_result}
