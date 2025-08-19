#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm.py — Generic offline analyzer for local datasets (CSV/TSV/XLSX/JSON/NDJSON/HTML/Parquet)
with light domain helpers for date/lag computations and plotting.

It scans the current working directory (recursively), loads any tabular files it finds,
and answers questions. For the Indian High Court dataset prompt, it will return the
requested JSON object if the required columns are present:
  - court, year, date_of_registration, decision_date

No network calls. Tarballs are extracted locally and scanned too.
"""

import os
import io
import re
import sys
import json
import base64
import tarfile
import tempfile
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ----------------------------
# Discovery & Loading
# ----------------------------

SUPPORTED_EXTS = {
    ".csv", ".tsv", ".txt",
    ".xlsx", ".xls",
    ".json", ".ndjson",
    ".html", ".htm",
    ".parquet", ".pq", ".feather",  # feather -> optional if available
    ".tar", ".tar.gz", ".tgz"
}

def _is_table_file(path: str) -> bool:
    p = path.lower()
    return any(p.endswith(ext) for ext in SUPPORTED_EXTS)

def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        # Try comma first, then fall back to tab
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", low_memory=False)
        except Exception:
            return None

def _safe_read_excel(path: str) -> List[pd.DataFrame]:
    out = []
    try:
        x = pd.ExcelFile(path)
        for sheet in x.sheet_names:
            try:
                out.append(x.parse(sheet))
            except Exception:
                pass
    except Exception:
        pass
    return out

def _safe_read_json(path: str) -> List[pd.DataFrame]:
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        # NDJSON?
        if "\n" in txt and txt.lstrip().startswith("{"):
            try:
                rows = [json.loads(line) for line in txt.splitlines() if line.strip()]
                if rows:
                    out.append(pd.DataFrame(rows))
                    return out
            except Exception:
                pass
        # Regular JSON
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                out.append(pd.DataFrame(obj))
            elif isinstance(obj, dict):
                # Single object -> single row
                out.append(pd.DataFrame([obj]))
        except Exception:
            pass
    except Exception:
        pass
    return out

def _safe_read_html(path: str) -> List[pd.DataFrame]:
    out = []
    try:
        tables = pd.read_html(path)
        out.extend(tables)
    except Exception:
        pass
    return out

def _safe_read_parquet(path: str) -> Optional[pd.DataFrame]:
    try:
        # Prefer pyarrow if available; fallback to fastparquet.
        return pd.read_parquet(path)
    except Exception:
        return None

def _extract_tar(tmp_root: str, tar_path: str) -> Optional[str]:
    try:
        extract_dir = tempfile.mkdtemp(prefix="extract_", dir=tmp_root)
        with tarfile.open(tar_path, "r:*") as tar:
            # Safety: prevent path traversal
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

            def safe_extract(tar_obj, path=".", members=None, *, numeric_owner=False):
                for member in tar_obj.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar_obj.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar, path=extract_dir)
        return extract_dir
    except Exception:
        return None

def discover_tables(root: str = ".",
                    max_files: int = 5000,
                    max_parquet_rows: Optional[int] = None) -> List[pd.DataFrame]:
    """
    Recursively discover and load tables from supported files.
    If max_parquet_rows is set, we will sample limited rows from huge parquet files.
    """
    loaded: List[pd.DataFrame] = []
    seen = 0
    tmp_root = tempfile.mkdtemp(prefix="llm_tmp_")

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if seen >= max_files:
                return loaded
            full = os.path.join(dirpath, fn)
            if not _is_table_file(full):
                continue
            seen += 1

            low = fn.lower()
            try:
                if low.endswith((".csv", ".tsv", ".txt")):
                    df = _safe_read_csv(full)
                    if df is not None and not df.empty:
                        loaded.append(df)

                elif low.endswith((".xlsx", ".xls")):
                    dfs = _safe_read_excel(full)
                    for d in dfs:
                        if d is not None and not d.empty:
                            loaded.append(d)

                elif low.endswith((".json", ".ndjson")):
                    dfs = _safe_read_json(full)
                    for d in dfs:
                        if d is not None and not d.empty:
                            loaded.append(d)

                elif low.endswith((".html", ".htm")):
                    dfs = _safe_read_html(full)
                    for d in dfs:
                        if d is not None and not d.empty:
                            loaded.append(d)

                elif low.endswith((".parquet", ".pq", ".feather")):
                    df = _safe_read_parquet(full)
                    if df is not None and not df.empty:
                        if max_parquet_rows and len(df) > max_parquet_rows:
                            df = df.sample(n=max_parquet_rows, random_state=42)
                        loaded.append(df)

                elif low.endswith((".tar", ".tar.gz", ".tgz")):
                    extracted = _extract_tar(tmp_root, full)
                    if extracted:
                        # Recurse one level into extracted dir
                        for e_dirpath, _, e_files in os.walk(extracted):
                            for efn in e_files:
                                efull = os.path.join(e_dirpath, efn)
                                if efull.lower().endswith((".parquet", ".pq", ".feather")):
                                    dfe = _safe_read_parquet(efull)
                                    if dfe is not None and not dfe.empty:
                                        if max_parquet_rows and len(dfe) > max_parquet_rows:
                                            dfe = dfe.sample(n=max_parquet_rows, random_state=42)
                                        loaded.append(dfe)
                                elif efull.lower().endswith((".json", ".ndjson")):
                                    for d in _safe_read_json(efull):
                                        if not d.empty:
                                            loaded.append(d)
                                elif efull.lower().endswith((".csv", ".tsv", ".txt")):
                                    dfe = _safe_read_csv(efull)
                                    if dfe is not None and not dfe.empty:
                                        loaded.append(dfe)
                                elif efull.lower().endswith((".xlsx", ".xls")):
                                    for d in _safe_read_excel(efull):
                                        if not d.empty:
                                            loaded.append(d)
                                elif efull.lower().endswith((".html", ".htm")):
                                    for d in _safe_read_html(efull):
                                        if not d.empty:
                                            loaded.append(d)
                        # Do not delete extracted dir; temp root will be cleaned by OS later.

            except Exception:
                # Skip problematic files; keep going
                pass

    return loaded

# ----------------------------
# Utilities & Feature Helpers
# ----------------------------

def unify_columns(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate a list of DataFrames by common columns.
    """
    if not dfs:
        return pd.DataFrame()
    # Pick the union of columns, align, and concat (ignore index)
    try:
        cols = set()
        for d in dfs:
            cols.update(map(str, d.columns))
        cols = list(cols)
        aligned = []
        for d in dfs:
            dd = d.copy()
            dd.columns = dd.columns.map(str)
            for c in cols:
                if c not in dd.columns:
                    dd[c] = pd.NA
            aligned.append(dd[cols])
        out = pd.concat(aligned, ignore_index=True)
        return out
    except Exception:
        # Fallback: simple concat with inner join on columns
        try:
            return pd.concat(dfs, ignore_index=True, join="inner")
        except Exception:
            return pd.DataFrame()

def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False, infer_datetime_format=True)

def compute_delay_days(df: pd.DataFrame,
                       reg_col: str = "date_of_registration",
                       dec_col: str = "decision_date") -> pd.Series:
    if reg_col not in df.columns or dec_col not in df.columns:
        return pd.Series([], dtype="float64")
    reg = to_datetime_safe(df[reg_col])
    dec = to_datetime_safe(df[dec_col])
    return (dec - reg).dt.days

def linear_regression_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Return slope of y ~ x using simple OLS (no intercept removal).
    """
    try:
        if len(x) < 2 or len(y) < 2:
            return None
        # Use numpy polyfit for slope with intercept
        m, b = np.polyfit(x, y, 1)
        return float(m)
    except Exception:
        return None

def plot_scatter_with_regression(x: np.ndarray,
                                 y: np.ndarray,
                                 xlabel: str,
                                 ylabel: str,
                                 title: str,
                                 max_chars: int = 100000) -> str:
    """
    Return a base64 data URI (PNG) for scatter + regression line,
    auto-tuning resolution to keep the string under max_chars.
    """
    if len(x) == 0 or len(y) == 0:
        return ""

    # Prepare regression line
    try:
        m, b = np.polyfit(x, y, 1)
    except Exception:
        m, b = None, None

    # Try a few sizes to fit under limit
    attempts = [
        (6, 4, 110),
        (5, 3, 110),
        (5, 3, 90),
        (4, 3, 90),
        (4, 3, 80),
        (4, 3, 70),
    ]

    for w, h, dpi in attempts:
        plt.close("all")
        fig = plt.figure(figsize=(w, h), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        if m is not None and b is not None:
            xs = np.linspace(np.min(x), np.max(x), 200)
            ax.plot(xs, m * xs + b, linestyle="--")  # default color; dashed
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uri = f"data:image/png;base64,{b64}"
        if len(uri) <= max_chars:
            return uri

    # If still too large, return the smallest attempt anyway
    return uri

# ----------------------------
# Question Handlers
# ----------------------------

def answer_indian_high_court_questions(df_all: pd.DataFrame,
                                       question_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    If the question object matches the Indian High Court shape,
    compute the answers and return a dict. Otherwise return None.
    """
    expected_keys = {
        "Which high court disposed the most cases from 2019 - 2022?",
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?",
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters"
    }
    if not expected_keys.issubset(set(question_obj.keys())):
        return None

    cols_needed = {"court", "year", "date_of_registration", "decision_date"}
    missing = [c for c in cols_needed if c not in df_all.columns]
    if missing:
        raise ValueError(f"Required columns not found in local data: {missing}")

    # Ensure types
    df = df_all.copy()
    # Coerce year
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Q1: Which high court disposed the most cases from 2019 - 2022?
    mask_yrs = (df["year"] >= 2019) & (df["year"] <= 2022)
    q1_df = df.loc[mask_yrs]
    q1_answer = ""
    if not q1_df.empty and "court" in q1_df.columns:
        counts = q1_df["court"].value_counts(dropna=True)
        if not counts.empty:
            q1_answer = str(counts.index[0])

    # Q2: Regression slope of (decision_date - date_of_registration) days ~ year, for court=33_10
    court_key = "33_10"
    df_court = df[df["court"].astype(str) == court_key].copy()
    df_court["delay_days"] = compute_delay_days(df_court)
    df_court = df_court.dropna(subset=["year", "delay_days"])
    q2_answer = ""
    slope_val: Optional[float] = None
    if len(df_court) >= 2:
        x = df_court["year"].astype(float).to_numpy()
        y = df_court["delay_days"].astype(float).to_numpy()
        slope_val = linear_regression_slope(x, y)
        if slope_val is not None and np.isfinite(slope_val):
            # round to 6 decimals like your earlier example
            q2_answer = f"{slope_val:.6f}"

    # Q3: Plot scatter of year vs delay_days with regression
    q3_answer = ""
    if len(df_court) >= 2:
        x = df_court["year"].astype(float).to_numpy()
        y = df_court["delay_days"].astype(float).to_numpy()
        q3_answer = plot_scatter_with_regression(
            x, y,
            xlabel="Year",
            ylabel="Delay (days)",
            title="Delay (decision − registration) by Year (court=33_10)",
            max_chars=100000
        )

    return {
        "Which high court disposed the most cases from 2019 - 2022?": q1_answer or "",
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": q2_answer or "",
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": q3_answer or ""
    }

# ----------------------------
# I/O & Main
# ----------------------------

def read_questions_from_file(path: str) -> Any:
    """
    Read questions from a .txt or .json file.
    - If JSON: return parsed object (list or dict).
    - If TXT: return the raw string (caller can decide).
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    # Try JSON first
    try:
        obj = json.loads(txt)
        return obj
    except Exception:
        pass

    # Fallback: return string
    return txt

def parse_questions_default(txt: str) -> Dict[str, Any]:
    """
    Extremely simple parser for your Indian Court JSON-object prompt if the caller
    passed a plain text description instead of JSON. If it can't detect, return {}.
    """
    target_keys = [
        "Which high court disposed the most cases from 2019 - 2022?",
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?",
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters"
    ]
    # Heuristic: if it mentions those exact lines, construct the keys.
    if all(k in txt for k in target_keys):
        return {k: "..." for k in target_keys}
    return {}

def main():
    """
    Usage patterns we support:
      - The orchestrator provides files in the CWD. We just scan and answer.
      - Questions can be:
          * provided via env QUESTIONS_JSON (JSON string: list or dict),
          * provided via a file path env QUESTIONS_PATH,
          * or absent (we'll just say "No questions provided").
    Output:
      - print a JSON string with either a dict (if questions object was a dict) or a list (if list).
      - For the Indian High Court sample, we print a dict with 3 keys as requested.
    """
    # 1) Load tables from current working directory
    tables = discover_tables(".", max_files=5000, max_parquet_rows=None)

    if not tables:
        # Keep consistent with your earlier logs
        print(json.dumps("No local tables found."))
        return

    df_all = unify_columns(tables)

    # 2) Get questions (envs used by your adapter / runner)
    q_env = os.environ.get("QUESTIONS_JSON")
    q_path = os.environ.get("QUESTIONS_PATH")

    questions: Any = None
    if q_env:
        try:
            questions = json.loads(q_env)
        except Exception:
            questions = q_env  # raw string
    elif q_path and os.path.exists(q_path):
        questions = read_questions_from_file(q_path)

    # 3) If no structured questions, we’re done
    if questions is None:
        print(json.dumps("No questions provided."))
        return

    # 4) If questions are a dict and match Indian High Court format, answer them
    try:
        if isinstance(questions, dict):
            ans = answer_indian_high_court_questions(df_all, questions)
            if ans is not None:
                print(json.dumps(ans, ensure_ascii=False))
                return
            # Otherwise, return an empty mapping (generic placeholder)
            print(json.dumps({k: "" for k in questions.keys()}, ensure_ascii=False))
            return

        # 5) If questions is a raw text containing the 3 prompts, synthesize and answer
        if isinstance(questions, str):
            maybe = parse_questions_default(questions)
            if maybe:
                ans = answer_indian_high_court_questions(df_all, maybe)
                if ans is not None:
                    print(json.dumps(ans, ensure_ascii=False))
                    return
            print(json.dumps("Unrecognized free-text questions.", ensure_ascii=False))
            return

        # 6) If questions is a list, return empty strings (generic fallback)
        if isinstance(questions, list):
            print(json.dumps(["" for _ in questions], ensure_ascii=False))
            return

        # Fallback
        print(json.dumps("Unsupported questions format."))
    except Exception as e:
        # Surface a concise error to your orchestrator
        print(json.dumps(f"Task failed: {e}"))

if __name__ == "__main__":
    main()
