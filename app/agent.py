# agent.py
from . import llm
from . import sandbox
import json
import re
from typing import Dict, Tuple, List
from urllib.request import Request, urlopen
import os

# Optional S3 support
try:
    import boto3
    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False

MAX_REFINEMENT_ATTEMPTS = 2

URL_PATTERN = re.compile(r"""(?P<url>https?://[^\s<>'"()]+)""", re.IGNORECASE)
S3_PATTERN = re.compile(r"""s3://([^/]+)/(.+)""")

def _extract_urls(text: str) -> List[str]:
    return [m.group("url") for m in URL_PATTERN.finditer(text or "")]

def _extract_s3_paths(text: str) -> List[Tuple[str, str]]:
    return [m.groups() for m in S3_PATTERN.finditer(text or "")]

def _fetch_url(url: str) -> Tuple[bytes, str]:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=20) as resp:
            content = resp.read()
            name = url.split("/")[-1] or "page.html"
            if not name.endswith(".html"):
                name += ".html"
            return content, name
    except Exception as e:
        fallback = f"<html><body><pre>FETCH_ERROR: {e}</pre></body></html>".encode("utf-8")
        return fallback, "fetch_error.html"

def _fetch_s3(bucket: str, key: str) -> Tuple[bytes, str]:
    if not _HAS_BOTO3:
        return b"S3 support not available", f"{bucket}_{os.path.basename(key)}.s3stub"
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read()
        fname = os.path.basename(key) or f"{bucket}_object"
        return content, fname
    except Exception as e:
        return str(e).encode("utf-8"), f"{bucket}_{os.path.basename(key)}.s3err"

async def run_data_analyst_agent(question_content_str: str, files: Dict[str, bytes]) -> dict:
    """
    Orchestrator for the data analyst agent.
    """
    attached_files: Dict[str, bytes] = dict(files or {})

    # 1) URLs embedded in the prompt → fetch and attach as local HTML
    urls = _extract_urls(question_content_str)
    for idx, u in enumerate(urls, start=1):
        content, name = _fetch_url(u)
        fname = f"url_{idx}_{name}"
        attached_files[fname] = content

    # 2) S3 paths in the prompt (optional)
    s3_refs = _extract_s3_paths(question_content_str)
    for idx, (bucket, key) in enumerate(s3_refs, start=1):
        content, name = _fetch_s3(bucket, key)
        fname = f"s3_{idx}_{name}"
        attached_files[fname] = content

    # 3) Data-bearing filenames (exclude questions.txt if present)
    data_filenames = [f for f in attached_files.keys() if not f.endswith("questions.txt")]

    # 4) If no files → fallback to direct LLM (best-effort)
    if not data_filenames:
        try:
            answers = llm.answer_questions_directly(question_content_str)
            return {"answers": answers}
        except Exception as e:
            return {"error": f"Direct answering failed: {e}"}

    # 5) Build file-type hints for the code generator
    hints = []
    for fname in data_filenames:
        lower = fname.lower()
        if lower.endswith(".html"):
            hints.append(f"File {fname} is HTML → parse with pandas.read_html + get_relevant_table()")
        elif lower.endswith(".csv"):
            hints.append(f"File {fname} is CSV → use pandas.read_csv")
        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            hints.append(f"File {fname} is Excel → use pandas.read_excel")
        elif lower.endswith(".geojson") or lower.endswith(".shp"):
            hints.append(f"File {fname} is geospatial → use geopandas.read_file")
        elif lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg"):
            hints.append(f"File {fname} is an image → use Pillow (PIL.Image)")
        elif lower.endswith(".json"):
            hints.append(f"File {fname} is JSON → use pandas.read_json or json.load")

    context = (
        f"User request: {question_content_str}\n"
        f"Available local files: {', '.join(data_filenames)}\n"
        f"File type hints: {'; '.join(hints) if hints else 'None'}\n"
        f"Notes: Internet access is not available inside the sandbox. "
        f"All analysis must be done on the provided local files only."
    )

    # 6) Plan (one deterministic step when there is a fetched URL/S3 object)
    if urls or s3_refs:
        plan = [
            "Load and parse the provided local files appropriately based on file type, "
            "compute the answers to all questions, and print the final JSON array. "
            "For any requested plots, output as base64 PNG data URIs under 100000 bytes. "
            "Do not make any network calls; only read local files."
        ]
    else:
        plan = llm.get_plan_from_llm(question_content_str, data_filenames)

    if not plan or "Error" in plan[0]:
        return {"error": "Could not generate a plan for the request."}

    # 7) Execute the plan with refinement
    final_result = ""
    for task in plan:
        code_to_run = llm.get_code_from_llm(task, context)

        for attempt in range(MAX_REFINEMENT_ATTEMPTS):
            stdout, stderr = sandbox.run_code_in_sandbox(code_to_run, attached_files)

            if stderr:
                # Quick, user-friendly fallbacks
                if "KeyError" in stderr or "No suitable" in stderr:
                    return {"answers": ["No suitable column found in table."]}

                if "could not convert string to float" in stderr:
                    # Force the generator to use clean_numeric()
                    refinement_context = (
                        f"{context}\n\nPrevious code failed due to numeric parsing.\n"
                        f"Error:\n{stderr}\n\n"
                        f"⚠️ Retry and **use clean_numeric(series)** for all numeric conversions. "
                        f"Do not use .astype(float) on raw strings."
                    )
                    code_to_run = llm.get_code_from_llm(task, refinement_context)
                    continue

                # Generic refine
                refinement_context = (
                    f"{context}\n\nThe previous code attempt failed. Please fix it.\n"
                    f"Error:\n{stderr}"
                )
                code_to_run = llm.get_code_from_llm(task, refinement_context)

                if attempt == MAX_REFINEMENT_ATTEMPTS - 1:
                    return {"error": f"Task failed after multiple attempts: {task}", "details": stderr}
            else:
                context += f"\n\nResult of '{task}':\n{stdout}"
                final_result = stdout
                break

    # 8) Parse/return the final JSON
    try:
        return json.loads(final_result)
    except json.JSONDecodeError:
        return {"result": final_result}
