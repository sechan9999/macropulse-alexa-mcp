"""Where each stage writes its results (validation/results/stageN/), as JSON and CSV."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from validation.data import ROOT

RESULTS = ROOT / "validation" / "results"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 10)
    if isinstance(o, (pd.Timestamp,)):
        return o.date().isoformat()
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def save_json(stage: str, name: str, obj) -> Path:
    p = RESULTS / stage / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(obj), indent=1, sort_keys=True, ensure_ascii=False))
    return p


def save_csv(stage: str, name: str, df: pd.DataFrame) -> Path:
    p = RESULTS / stage / f"{name}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, lineterminator="\n", float_format="%.8g")
    return p


def load_json(stage: str, name: str):
    p = RESULTS / stage / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None
