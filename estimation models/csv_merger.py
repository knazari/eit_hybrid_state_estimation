import re
import numpy as np
import pandas as pd
from pathlib import Path

# ---- EDIT THESE ----
inputs = [
    {"path": "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/press_polar_steps_20250820_130254.csv", "probe_type": "small",  "trim_head": 2, "trim_tail": 2},
    {"path": "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/press_polar_steps_20250820_141938.csv", "probe_type": "small",  "trim_head": 2, "trim_tail": 2},
    {"path": "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/press_polar_steps_20250820_121312.csv", "probe_type": "medium", "trim_head": 2, "trim_tail": 2},
    {"path": "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/press_polar_steps_20250820_152653.csv", "probe_type": "medium", "trim_head": 2, "trim_tail": 2},
    {"path": "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/press_polar_steps_20250820_113814.csv", "probe_type": "large",  "trim_head": 2, "trim_tail": 2},
    {"path": "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/press_polar_steps_20250820_162339.csv", "probe_type": "large",  "trim_head": 2, "trim_tail": 2},
]

out_csv = Path("data/merged_with_probe.csv")
out_parquet = Path("data/merged_with_probe.parquet")

EIT_RE = re.compile(r"^eit_(\d+)(?:\.\d+)?$")  # eit_137, eit_137.1, ...

def trim_head_tail(df: pd.DataFrame, head: int, tail: int) -> pd.DataFrame:
    n = len(df)
    start = max(0, head or 0)
    end = n - (tail or 0)
    end = max(start, end)
    return df.iloc[start:end].copy()

def standardize_eit_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, collapse duplicates by mean, ensure uniqueness."""
    # 1) normalize whitespace and types
    df = df.rename(columns=lambda c: str(c).strip())

    # 2) rename eit_xxx.y → eit_xxx
    rename_map = {}
    for c in df.columns:
        m = EIT_RE.match(c)
        if m:
            rename_map[c] = f"eit_{m.group(1)}"
    if rename_map:
        df = df.rename(columns=rename_map)

    # 3) merge duplicates by row-wise mean (before dropping dups)
    counts = df.columns.value_counts()
    dup_names = counts[counts > 1].index.tolist()
    for name in dup_names:
        cols = [c for c in df.columns if c == name]
        # compute row-wise mean (numeric), ignoring NaN
        df[name] = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

    # 4) drop duplicate columns keeping the first occurrence
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df

def load_one(path: str, probe_type: str, trim_head: int, trim_tail: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = trim_head_tail(df, trim_head, trim_tail)
    df = standardize_eit_headers(df)
    df["probe_type"] = probe_type
    return df

# ---------- Load all ----------
dfs = []
for spec in inputs:
    dfs.append(load_one(spec["path"], spec["probe_type"], spec["trim_head"], spec["trim_tail"]))

# ---------- Build a stable column order ----------
# Use first file as reference for non-EIT columns, then append all EIT columns sorted by numeric index.
ref = dfs[0]
non_eit_cols = [c for c in ref.columns if not c.startswith("eit_") and c != "probe_type"]
# collect union of EIT columns across all files
eit_union = set()
for d in dfs:
    eit_union.update([c for c in d.columns if c.startswith("eit_")])
def eit_index(c):
    m = EIT_RE.match(c)
    return int(m.group(1)) if m else 10**9
eit_cols_sorted = sorted(eit_union, key=eit_index)

# final order: metadata (from first file, in order) → all EIT sorted → probe_type
ref_order = non_eit_cols + eit_cols_sorted + ["probe_type"]

# ---------- Align all frames to this order ----------
aligned = []
for d in dfs:
    # Add any missing columns as NaN, then reorder
    missing = [c for c in ref_order if c not in d.columns]
    for c in missing:
        d[c] = np.nan
    d = d[ref_order]
    # Coerce EIT columns to numeric (after alignment)
    for c in eit_cols_sorted:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    aligned.append(d)

# ---------- Concatenate ----------
merged = pd.concat(aligned, ignore_index=True)

# If you want to sort by time (uncomment if your 't' column exists and is ISO-like):
# if "t" in merged.columns:
#     merged["t"] = pd.to_datetime(merged["t"], errors="coerce")
#     merged = merged.sort_values("t", kind="stable").reset_index(drop=True)

# ---------- Save ----------
out_csv.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_csv, index=False)
try:
    merged.to_parquet(out_parquet, index=False)
except Exception as e:
    print(f"[warn] Parquet not written ({e}); install pyarrow or fastparquet)")

print(f"✅ Merged {len(inputs)} files → {out_csv} | rows={len(merged)} | cols={len(merged.columns)}")
