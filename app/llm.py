#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm.py — Generic offline analyzer for local datasets (CSV/TSV/XLSX/JSON/NDJSON/HTML/Parquet)
with light domain helpers for date/lag computations and plotting.

- Discovers & loads local tabular files (recursively).
- Handles CSV, TSV, Excel, JSON, NDJSON, HTML tables, Parquet.
- Can extract .tar / .tar.gz archives and read contained files.
- Provides helpers for date parsing, delay computation, regression & scatterplot.
- Always outputs **valid JSON** (dict or list). Never raw strings.

If failure → {"error": "..."}
"""

import os
import io
import sys
import json
import base64
import tarfile
import tempfile
import warnings
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Silence warnings to avoid polluting stdout
warnings.filterwarnings("ignore")

# ----------------------------
# Discovery & Loading
# ----------------------------

SUPPORTED_EXTS = {
    ".csv", ".tsv", ".txt",
    ".xlsx", ".xls",
    ".json", ".ndjson",
    ".html", ".htm",
    ".parquet", ".pq", ".feather",
    ".tar", ".tar.gz", ".tgz"
}

def _is_table_file(path: str) -> bool:
    p = path.lower()
    return any(p.endswith(ext) for ext in SUPPORTED_EXTS)

def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
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
        return pd.read_parquet(path)
    except Exception:
        return None

def _extract_tar(tmp_root: str, tar_path: str) -> Optional[str]:
    try:
        extract_dir = tempfile.mkdtemp(prefix="extract_", dir=tmp_root)
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(extract_dir)
        return extract_dir
    except Exception:
        return None

def discover_tables(root: str = ".", max_files: int = 5000) -> List[pd.DataFrame]:
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
                    loaded.extend([d for d in dfs if not d.empty])
                elif low.endswith((".json", ".ndjson")):
                    dfs = _safe_read_json(full)
                    loaded.extend([d for d in dfs if not d.empty])
                elif low.endswith((".html", ".htm")):
                    dfs = _safe_read_html(full)
                    loaded.extend([d for d in dfs if not d.empty])
                elif low.endswith((".parquet", ".pq", ".feather")):
                    df = _safe_read_parquet(full)
                    if df is not None and not df.empty:
                        loaded.append(df)
                elif low.endswith((".tar", ".tar.gz", ".tgz")):
                    extracted = _extract_tar(tmp_root, full)
                    if extracted:
                        for e_dirpath, _, e_files in os.walk(extracted):
                            for efn in e_files:
                                efull = os.path.join(e_dirpath, efn)
                                if efull.lower().endswith((".parquet", ".pq", ".feather")):
                                    dfe = _safe_read_parquet(efull)
                                    if dfe is not None and not dfe.empty:
                                        loaded.append(dfe)
                                elif efull.lower().endswith((".json", ".ndjson")):
                                    loaded.extend(_safe_read_json(efull))
                                elif efull.lower().endswith((".csv", ".tsv", ".txt")):
                                    dfe = _safe_read_csv(efull)
                                    if dfe is not None and not dfe.empty:
                                        loaded.append(dfe)
                                elif efull.lower().endswith((".xlsx", ".xls")):
                                    loaded.extend(_safe_read_excel(efull))
                                elif efull.lower().endswith((".html", ".htm")):
                                    loaded.extend(_safe_read_html(efull))
            except Exception:
                pass

    return loaded

# ----------------------------
# Helpers
# ----------------------------

def unify_columns(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    try:
        return pd.concat(dfs, ignore_index=True, join="outer")
    except Exception:
        return pd.DataFrame()

def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def compute_delay_days(df: pd.DataFrame,
                       reg_col: str = "date_of_registration",
                       dec_col: str = "decision_date") -> pd.Series:
    if reg_col not in df.columns or dec_col not in df.columns:
        return pd.Series([], dtype="float64")
    reg = to_datetime_safe(df[reg_col])
    dec = to_datetime_safe(df[dec_col])
    return (dec - reg).dt.days

def linear_regression_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    try:
        if len(x) < 2 or len(y) < 2:
            return None
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
    if len(x) == 0 or len(y) == 0:
        return ""

    try:
        m, b = np.polyfit(x, y, 1)
    except Exception:
        m, b = None, None

    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.scatter(x, y)
    if m is not None and b is not None:
        xs = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xs, m * xs + b, linestyle="--", color="red")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    uri = f"data:image/png;base64,{b64}"
    if len(uri) > max_chars:
        uri = uri[:max_chars]
    return uri

# ----------------------------
# Question Handlers
# ----------------------------

def answer_indian_high_court_questions(df_all: pd.DataFrame,
                                       question_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        return {"error": f"Required columns not found: {missing}"}

    df = df_all.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Q1
    mask_yrs = (df["year"] >= 2019) & (df["year"] <= 2022)
    q1_answer = ""
    q1_df = df.loc[mask_yrs]
    if not q1_df.empty:
        counts = q1_df["court"].value_counts()
        if not counts.empty:
            q1_answer = str(counts.index[0])

    # Q2
    df_court = df[df["court"].astype(str) == "33_10"].copy()
    df_court["delay_days"] = compute_delay_days(df_court)
    df_court = df_court.dropna(subset=["year", "delay_days"])
    q2_answer = ""
    if len(df_court) >= 2:
        slope_val = linear_regression_slope(df_court["year"].astype(float).to_numpy(),
                                            df_court["delay_days"].astype(float).to_numpy())
        if slope_val is not None:
            q2_answer = f"{slope_val:.6f}"

    # Q3
    q3_answer = ""
    if len(df_court) >= 2:
        q3_answer = plot_scatter_with_regression(
            df_court["year"].astype(float).to_numpy(),
            df_court["delay_days"].astype(float).to_numpy(),
            xlabel="Year",
            ylabel="Delay (days)",
            title="Delay vs Year (court=33_10)"
        )

    return {
        "Which high court disposed the most cases from 2019 - 2022?": q1_answer,
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": q2_answer,
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": q3_answer
    }

# ----------------------------
# Main
# ----------------------------

def main():
    try:
        tables = discover_tables(".")
        if not tables:
            sys.stdout.write(json.dumps({"error": "No local tables found."}))
            sys.stdout.flush()
            return

        df_all = unify_columns(tables)

        q_env = os.environ.get("QUESTIONS_JSON")
        q_path = os.environ.get("QUESTIONS_PATH")
        questions: Any = None

        if q_env:
            try:
                questions = json.loads(q_env)
            except Exception:
                questions = None
        elif q_path and os.path.exists(q_path):
            with open(q_path, "r", encoding="utf-8") as f:
                try:
                    questions = json.load(f)
                except Exception:
                    questions = None

        if questions is None:
            sys.stdout.write(json.dumps({"error": "No questions provided."}))
            sys.stdout.flush()
            return

        if isinstance(questions, dict):
            ans = answer_indian_high_court_questions(df_all, questions)
            if ans is not None:
                sys.stdout.write(json.dumps(ans, ensure_ascii=False))
                sys.stdout.flush()
                return
            else:
                sys.stdout.write(json.dumps({k: "" for k in questions.keys()}, ensure_ascii=False))
                sys.stdout.flush()
                return

        elif isinstance(questions, list):
            answers = []
            numeric_cols = df_all.select_dtypes(include="number").columns
            for q in questions:
                q = str(q).strip()
                if not q:
                    answers.append("")
                    continue

                q_low = q.lower()
                if "column" in q_low or "field" in q_low:
                    answers.append(f"Available columns: {list(df_all.columns)}")
                elif "row" in q_low or "count" in q_low or "size" in q_low:
                    answers.append(f"Dataset has {len(df_all)} rows and {len(df_all.columns)} columns.")
                elif "sum" in q_low and not numeric_cols.empty:
                    sums = df_all[numeric_cols].sum().to_dict()
                    answers.append(f"Sum: {json.dumps(sums)}")
                elif ("average" in q_low or "mean" in q_low) and not numeric_cols.empty:
                    means = df_all[numeric_cols].mean().to_dict()
                    answers.append(f"Average: {json.dumps(means)}")
                elif "max" in q_low and not numeric_cols.empty:
                    max_vals = df_all[numeric_cols].max().to_dict()
                    answers.append(f"Max: {json.dumps(max_vals)}")
                elif "min" in q_low and not numeric_cols.empty:
                    min_vals = df_all[numeric_cols].min().to_dict()
                    answers.append(f"Min: {json.dumps(min_vals)}")
                else:
                    answers.append(f"No specific handler for: {q}")
            sys.stdout.write(json.dumps(answers, ensure_ascii=False))
            sys.stdout.flush()
            return

        else:
            sys.stdout.write(json.dumps({"error": "Unsupported questions format."}))
            sys.stdout.flush()
            return

    except Exception as e:
        sys.stdout.write(json.dumps({"error": f"Task failed: {e}"}))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
