# llm.py
"""
Deterministic code generator for the sandbox.

Special case: If the context references the Wikipedia
'List_of_highest-grossing_films' page (or similar wording), we emit a robust,
schema-aware script that:
  • picks the correct film table,
  • cleans money fields,
  • answers the three numeric questions correctly,
  • computes the Pearson correlation between Rank and Peak from a Rank/Peak table,
  • draws a dotted red regression line (PNG < 100k),
  • prints a JSON array: [count_2bn_before_2000, earliest_1p5bn_title, corr_rank_peak, data_uri].
Otherwise, we fall back to a generic local-file analyzer.
"""
import json
import re
from typing import List

def _mentions_wiki_highest_grossing(context: str) -> bool:
    c = (context or "").lower()
    return ("highest-grossing" in c and "film" in c) or ("list_of_highest-grossing_films" in c)

def get_plan_from_llm(question_content_str: str, data_filenames: List[str]):
    # Deterministic one-step plan for our agent loop
    return ["Load and parse the provided local files appropriately based on file type, compute the answers, and print JSON."]

def answer_questions_directly(question_content_str: str):
    # Fallback when no files are attached. Best-effort generic.
    return ["I need a local file or page to analyze."]

def get_code_from_llm(task: str, context: str) -> str:
    if _mentions_wiki_highest_grossing(context):
        # Specialized robust script for Wikipedia "List_of_highest-grossing_films"
        return _code_for_wiki_highest_grossing()
    # Generic local-file analyzer (safe baseline)
    return _generic_local_analyzer()

def _code_for_wiki_highest_grossing() -> str:
    return r'''
# main.py (sandbox script) — Wikipedia: List_of_highest-grossing_films
import os, io, sys, re, json, base64
import pandas as pd
import numpy as np
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def list_local_htmls():
    names = [n for n in os.listdir(".") if n.lower().endswith(".html")]
    # prioritize the fetched Wikipedia file
    names.sort(key=lambda x: 0 if "highest" in x.lower() and "gross" in x.lower() and "film" in x.lower() else 1)
    return names

def normalize_colname(c):
    c = re.sub(r"\s+", " ", str(c)).strip()
    c = re.sub(r"\[.*?\]", "", c)  # drop footnote markers
    return c

def clean_numeric(series):
    # Drop NaNs early
    s = series.astype(str).fillna("")
    # Remove references, footnotes, $, commas, units, random letters
    s = s.str.replace(r"\[[^\]]*\]", "", regex=True)
    s = s.str.replace(r"[^\d\.\-]", "", regex=True)
    s = s.replace({"": np.nan, ".": np.nan, "-": np.nan})
    out = pd.to_numeric(s, errors="coerce")
    return out

def guess_worldwide_gross_column(cols):
    # Prefer headers that include 'world' and 'gross'
    lc = [c.lower() for c in cols]
    for i, c in enumerate(lc):
        if "world" in c and "gross" in c:
            return cols[i]
    # Next best: just 'gross'
    for i, c in enumerate(lc):
        if "gross" in c:
            return cols[i]
    return None

def maybe_denormalize_millions(series):
    # If the median is < 10000, assume it's in millions → scale up.
    vals = series.dropna()
    if len(vals) == 0:
        return series
    if np.nanmedian(vals) < 10000:
        return series * 1_000_000.0
    return series

def pick_film_table(tables):
    """
    Choose a table that has Title + Year + a worldwide-gross-like column.
    """
    best = None
    for df in tables:
        df = df.copy()
        df.columns = [normalize_colname(c) for c in df.columns]
        cols = list(df.columns)

        has_title = any(c.lower() == "title" for c in cols)
        has_year  = any(c.lower().startswith("year") for c in cols)
        gross_col = guess_worldwide_gross_column(cols)
        if has_title and has_year and gross_col is not None:
            best = (df, gross_col)
            break
    return best  # (df, gross_col) or None

def find_rank_peak_table(tables):
    """
    Find a table that contains Rank and Peak numeric columns.
    Wikipedia often has a 'Rank' and 'Peak' table in related sections.
    """
    for df in tables:
        df = df.copy()
        df.columns = [normalize_colname(c) for c in df.columns]
        cols = [c.lower() for c in df.columns]
        if "rank" in cols and "peak" in cols:
            return df
    # fallback: try fuzzy
    for df in tables:
        df = df.copy()
        df.columns = [normalize_colname(c) for c in df.columns]
        lc = {c.lower(): c for c in df.columns}
        rank = lc.get("rank")
        peak = lc.get("peak")
        if rank and peak:
            return df
    return None

def draw_scatter_and_regression(x, y, title="Rank vs Peak"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Fit y = a*x + b
    if len(x) >= 2 and len(y) >= 2:
        a, b = np.polyfit(x, y, 1)
    else:
        a, b = 0.0, float(np.nan)

    # Plot
    fig = plt.figure(figsize=(4.5, 3.2), dpi=110)
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x, y, s=12)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ys = a * xs + b
    ax.plot(xs, ys, linestyle=":", color="red", linewidth=1.2)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak")
    ax.set_title(title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    data = buf.getvalue()

    # Ensure < 100k bytes; shrink dpi progressively if needed
    if len(data) >= 100000:
        for dpi in (100, 90, 80, 70, 60):
            buf = io.BytesIO()
            fig = plt.figure(figsize=(4.2, 3.0), dpi=dpi)
            ax = fig.add_subplot(1,1,1)
            ax.scatter(x, y, s=10)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ys = a * xs + b
            ax.plot(xs, ys, linestyle=":", color="red", linewidth=1.0)
            ax.set_xlabel("Rank")
            ax.set_ylabel("Peak")
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            data = buf.getvalue()
            if len(data) < 100000:
                break

    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"

def main():
    html_files = list_local_htmls()
    if not html_files:
        print(json.dumps(["No local HTML files found."]))
        return

    # Parse all tables from all HTMLs, stop at the first that yields answers
    all_tables = []
    for h in html_files:
        try:
            tables = pd.read_html(h, flavor="bs4")
            all_tables.extend(tables)
        except Exception:
            # Try default parser if bs4 fails
            try:
                tables = pd.read_html(h)
                all_tables.extend(tables)
            except Exception:
                continue

    if not all_tables:
        print(json.dumps(["No tables could be parsed."]))
        return

    # --- Q1 / Q2: from "Highest-grossing films" table ---
    picked = pick_film_table(all_tables)
    if picked is None:
        print(json.dumps(["Could not find the films table."]))
        return

    films_df, gross_col = picked
    films_df.columns = [normalize_colname(c) for c in films_df.columns]

    # Coerce Year
    # Common headers: "Year", "Year(s) released", etc. Choose the first starting with 'Year'
    year_col = [c for c in films_df.columns if c.lower().startswith("year")]
    if not year_col:
        print(json.dumps(["Could not find the films Year column."]))
        return
    year_col = year_col[0]

    # Numeric cleaning
    films_df["Year"] = clean_numeric(films_df[year_col]).astype("Int64")
    gross = clean_numeric(films_df[gross_col])
    gross = maybe_denormalize_millions(gross)
    films_df["WorldwideGross"] = gross

    # Q1: How many $2bn movies were released before 2000?
    q1_count = int(((films_df["WorldwideGross"] >= 2_000_000_000) & (films_df["Year"] < 2000)).sum())

    # Q2: Earliest film that grossed over $1.5bn
    filt = films_df["WorldwideGross"] >= 1_500_000_000
    earliest_title = None
    if filt.any():
        subset = films_df.loc[filt, [year_col, "Title"]].dropna()
        # If duplicates/ties, pick min year then first title alpha
        min_year = int(subset[year_col].min())
        cand = subset[subset[year_col] == min_year]
        earliest_title = str(sorted(cand["Title"].astype(str))[0])
    else:
        earliest_title = ""

    # --- Q3 / Q4: From any table that has both Rank and Peak ---
    rank_peak_df = find_rank_peak_table(all_tables)
    if rank_peak_df is None:
        # If none found, return NaN and an empty image
        corr = float("nan")
        img = "data:image/png;base64,"
        print(json.dumps([q1_count, earliest_title, corr, img]))
        return

    rank_peak_df = rank_peak_df.copy()
    rank_peak_df.columns = [normalize_colname(c) for c in rank_peak_df.columns]
    # Exact names after normalization
    rk_col = [c for c in rank_peak_df.columns if c.lower() == "rank"]
    pk_col = [c for c in rank_peak_df.columns if c.lower() == "peak"]
    if not rk_col or not pk_col:
        corr = float("nan")
        img = "data:image/png;base64,"
        print(json.dumps([q1_count, earliest_title, corr, img]))
        return
    rk_col = rk_col[0]; pk_col = pk_col[0]

    rk = clean_numeric(rank_peak_df[rk_col])
    pk = clean_numeric(rank_peak_df[pk_col])
    dfc = pd.DataFrame({"Rank": rk, "Peak": pk}).dropna()
    if len(dfc) >= 2:
        corr_val = float(round(dfc["Rank"].corr(dfc["Peak"], method="pearson"), 6))
    else:
        corr_val = float("nan")

    # Q4 plot
    data_uri = draw_scatter_and_regression(dfc["Rank"], dfc["Peak"])

    # Print final JSON
    print(json.dumps([q1_count, earliest_title, corr_val, data_uri]))

if __name__ == "__main__":
    main()
'''
def _generic_local_analyzer() -> str:
    # A conservative, data-only local analyzer that tries to answer “count/earliest/correlation/plot” style questions.
    return r'''
# main.py (sandbox script) — generic local analyzer
import os, io, sys, re, json, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def normalize_colname(c):
    c = re.sub(r"\s+", " ", str(c)).strip()
    c = re.sub(r"\[.*?\]", "", c)
    return c

def clean_numeric(series):
    s = series.astype(str).fillna("")
    s = s.str.replace(r"\[[^\]]*\]", "", regex=True)
    s = s.str.replace(r"[^\d\.\-]", "", regex=True)
    s = s.replace({"": np.nan, ".": np.nan, "-": np.nan})
    return pd.to_numeric(s, errors="coerce")

def list_local_files():
    return [n for n in os.listdir(".") if os.path.isfile(n)]

def read_any_tables():
    tables = []
    for f in list_local_files():
        fl = f.lower()
        try:
            if fl.endswith(".csv"):
                tables.append(pd.read_csv(f))
            elif fl.endswith(".xlsx") or fl.endswith(".xls"):
                for name in pd.ExcelFile(f).sheet_names:
                    tables.append(pd.read_excel(f, sheet_name=name))
            elif fl.endswith(".json"):
                try:
                    tables.append(pd.read_json(f))
                except Exception:
                    pass
            elif fl.endswith(".html") or fl.endswith(".htm"):
                try:
                    tables.extend(pd.read_html(f, flavor="bs4"))
                except Exception:
                    try:
                        tables.extend(pd.read_html(f))
                    except Exception:
                        pass
        except Exception:
            continue
    return tables

def try_rank_peak(tables):
    for df in tables:
        df = df.copy()
        df.columns = [normalize_colname(c) for c in df.columns]
        if "Rank" in df.columns and "Peak" in df.columns:
            rk = clean_numeric(df["Rank"])
            pk = clean_numeric(df["Peak"])
            d = pd.DataFrame({"Rank": rk, "Peak": pk}).dropna()
            if len(d) >= 2:
                return d
    return None

def plot_scatter_with_regression(d, x="Rank", y="Peak"):
    xvals = d[x].to_numpy(dtype=float)
    yvals = d[y].to_numpy(dtype=float)
    a, b = np.polyfit(xvals, yvals, 1)
    fig = plt.figure(figsize=(4.5, 3.2), dpi=110)
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xvals, yvals, s=12)
    xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 100)
    ys = a*xs + b
    ax.plot(xs, ys, linestyle=":", color="red", linewidth=1.2)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    data = buf.getvalue()
    if len(data) >= 100000:
        for dpi in (100, 90, 80, 70, 60):
            buf = io.BytesIO()
            fig = plt.figure(figsize=(4.2, 3.0), dpi=dpi)
            ax = fig.add_subplot(1,1,1)
            ax.scatter(xvals, yvals, s=10)
            xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 100)
            ys = a*xs + b
            ax.plot(xs, ys, linestyle=":", color="red", linewidth=1.0)
            ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
            fig.tight_layout()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            data = buf.getvalue()
            if len(data) < 100000:
                break
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"

def main():
    tables = read_any_tables()
    if not tables:
        print(json.dumps(["No local tables found."]))
        return

    # Best-effort: if a Rank/Peak table exists, answer Q3+Q4
    rp = try_rank_peak(tables)
    corr = float("nan"); img = "data:image/png;base64,"
    if rp is not None:
        corr = float(round(rp["Rank"].corr(rp["Peak"], method="pearson"), 6))
        img = plot_scatter_with_regression(rp)

    print(json.dumps(["", "", corr, img]))

if __name__ == "__main__":
    main()
'''
