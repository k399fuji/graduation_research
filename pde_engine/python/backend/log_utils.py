# python/backend/log_utils.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import csv

# プロジェクトルート & ログディレクトリはここで一元管理
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LINE_WIDTH = 0.7  # Loss 曲線用のデフォルト線幅（GUI 側から参照）

@dataclass
class RunInfo:
    run_id: str


# -------- パス関連ヘルパ --------

def get_log_dir() -> Path:
    return LOG_DIR


def path_config(run_id: str) -> Path:
    return LOG_DIR / f"{run_id}_config.json"


def path_eval(run_id: str) -> Path:
    return LOG_DIR / f"{run_id}_eval.json"


def path_summary(run_id: str) -> Path:
    return LOG_DIR / f"{run_id}_summary.json"


def path_csv(run_id: str) -> Path:
    return LOG_DIR / f"{run_id}.csv"


def path_model(run_id: str) -> Path:
    return LOG_DIR / f"{run_id}_model.pt"


# -------- ロード系ユーティリティ --------

def load_config(run_id: str) -> Optional[Dict[str, Any]]:
    p = path_config(run_id)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_eval(run_id: str) -> Optional[Dict[str, Any]]:
    p = path_eval(run_id)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_summary(run_id: str) -> Optional[Dict[str, Any]]:
    p = path_summary(run_id)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_loss_csv(run_id: str) -> List[Dict[str, str]]:
    p = path_csv(run_id)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return []


# -------- run 一覧 --------

def list_runs() -> list[RunInfo]:
    """
    logs ディレクトリから run_id 一覧をざっくり拾う。
    *_config.json, *_summary.json, *.csv などから run_id を推定。
    """
    run_ids: set[str] = set()

    for p in LOG_DIR.glob("*.json"):
        name = p.name
        if name.endswith("_config.json"):
            run_ids.add(name[:-len("_config.json")])
        elif name.endswith("_summary.json"):
            run_ids.add(name[:-len("_summary.json")])

    for p in LOG_DIR.glob("*.csv"):
        # 例: heat1d_20251122_xxx.csv → stem 全体が run_id
        run_ids.add(p.stem)

    return [RunInfo(run_id=rid) for rid in sorted(run_ids)]
