"""SPY discovery: queued background batches and manual evidence/review commands.

Run ``python -m factory.discovery --help`` from the repository root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd
import yaml

from backtester.analytics.metrics import sortino_ratio
from backtester.config.loader import _from_dict
from backtester.config.validation import validate_run_config
from backtester.data.loader import load_symbol
from factory.generate import call_generator
from factory.results import read_records
from factory.settings_loader import load_settings
from factory.stages import run_backtest_stage, run_optimize_stage, run_wfo_stage
from factory.validate import validate_functional, validate_static

ROOT = Path(__file__).resolve().parent.parent
SEARCH_START, SEARCH_END = "2015-01-02", "2024-12-31"
REVIEW_START, REVIEW_END = "2025-01-01", "2026-05-14"
WINDOWS = {"enabled": True, "train_bars": 504, "test_bars": 126, "step_bars": 126}
log = logging.getLogger(__name__)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _snapshot(root: Path, start: str, end: str, name: str) -> dict:
    data = load_symbol("SPY", "csv", root / "data/raw", start, end)
    if data.empty or data.index.min() < pd.Timestamp(start) or data.index.max().strftime("%Y-%m-%d") != end:
        raise ValueError(f"SPY data unavailable for {start} through {end}")
    path = root / "discovery/data" / name / "SPY.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index_label="timestamp")
    return {"path": str(path.relative_to(root)), "sha256": _hash(path),
            "start": start, "end": end, "bars": len(data),
            "last_bar": data.index.max().strftime("%Y-%m-%d")}


def _valid_wfo_bundle(bundle: Path) -> bool:
    try:
        summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
        windows = json.loads((bundle / "window_results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(summary, dict) and isinstance(summary.get("oos_summary"), dict)
            and isinstance(summary.get("n_windows"), int)
            and summary["n_windows"] > 0
            and isinstance(windows, list) and len(windows) == summary["n_windows"])


def _evidence(root: Path, sid: str, rec: dict | None, config_hash: str, code_hash: str, data_hash: str) -> dict:
    backfill = root / "discovery/backfills" / f"{sid}.json"
    if backfill.exists():
        item = json.loads(backfill.read_text(encoding="utf-8"))
        if (item.get("config_sha256"), item.get("code_sha256"), item.get("data_sha256")) == (config_hash, code_hash, data_hash):
            bundle = _bundle(root, item["bundle"])
            if (_valid_wfo_bundle(bundle)
                    and item.get("summary_sha256") == _hash(bundle / "summary.json")
                    and item.get("windows_sha256") == _hash(bundle / "window_results.json")):
                return {"status": "available", "source": "backfill", **item}
    wfo = (rec or {}).get("wfo")
    if isinstance(wfo, dict):
        bundle = _bundle(root, wfo.get("run_bundle_path", ""))
        resolved = bundle / "config_resolved.yaml"
        run_config = yaml.safe_load(resolved.read_text(encoding="utf-8")) if resolved.is_file() else {}
        run_data = run_config.get("data", {}) if isinstance(run_config, dict) else {}
        if _valid_wfo_bundle(bundle) \
                and run_data.get("symbols") == ["SPY"] and run_data.get("start") == SEARCH_START \
                and run_data.get("end") == SEARCH_END:
            return {"status": "available", "source": "result_shard", "bundle": str(bundle),
                    "summary": json.loads((bundle / "summary.json").read_text(encoding="utf-8")),
                    "summary_sha256": _hash(bundle / "summary.json"),
                    "windows_sha256": _hash(bundle / "window_results.json")}
    return {"status": "missing", "reason": "no readable WFO summary and windows artifact"}


def _artifacts(root: Path, rec: dict | None) -> dict:
    out = {}
    for stage in ("backtest", "optimize", "wfo"):
        path = (rec or {}).get(stage) or {}
        bundle = _bundle(root, path.get("run_bundle_path", "")) if isinstance(path, dict) else root
        out[stage] = {"status": "available", "bundle": str(bundle)} if (bundle / "summary.json").is_file() else {"status": "missing"}
    return out


def index_library(root: Path = ROOT) -> dict:
    root = root.resolve()
    snapshot = _snapshot(root, SEARCH_START, SEARCH_END, "search")
    records = read_records(root / "factory/data/results")
    latest = {}
    for rec in sorted(records, key=lambda r: r.get("timestamp", "")):
        if rec.get("strategy_id"):
            latest[rec["strategy_id"]] = rec
    config_by_strategy: dict[str, list[Path]] = {}
    orphan_configs = []
    for path in sorted([*(root / "configs/wfo").glob("*.yaml"), *(root / "configs/backtests").glob("*.yaml")]):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        sid = cfg.get("strategy") if isinstance(cfg, dict) else None
        if isinstance(sid, str):
            config_by_strategy.setdefault(sid, []).append(path)
        else:
            orphan_configs.append(str(path.relative_to(root)))
    entries = {}
    for path in sorted((root / "strategies").glob("*.py")):
        if path.name == "__init__.py":
            continue
        sid = path.stem
        configs = config_by_strategy.get(sid, [])
        config = next((p for p in configs if p.parent.name == "wfo"), configs[0] if configs else None)
        rec = latest.get(sid)
        item = {"strategy": str(path.relative_to(root)), "code_sha256": _hash(path),
                "configs": [str(p.relative_to(root)) for p in configs],
                "result": rec, "artifacts": _artifacts(root, rec),
                "wfo": {"status": "missing", "reason": "no WFO config"}}
        if config:
            cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            item.update(config=str(config.relative_to(root)), config_sha256=_hash(config),
                        costs=cfg.get("execution", {}), data=cfg.get("data", {}))
            if cfg.get("wfo", {}).get("enabled"):
                item["wfo"] = _evidence(root, sid, rec, _hash(config), _hash(path), snapshot["sha256"])
        entries[sid] = item
    run_artifacts = {}
    for bundle in sorted((root / "output/runs").glob("*")):
        summary = bundle / "summary.json"
        if bundle.is_dir() and summary.is_file():
            run_artifacts[str(bundle.relative_to(root))] = {
                "summary_sha256": _hash(summary),
                "windows_sha256": _hash(bundle / "window_results.json") if (bundle / "window_results.json").is_file() else None,
            }
    manifest = {"search_data": snapshot, "source_data_sha256": _hash(root / "data/raw/SPY.csv"),
                "result_shards": {str(p.relative_to(root)): _hash(p) for p in sorted((root / "factory/data/results").glob("*.jsonl"))},
                "result_records": len(records), "run_artifacts": run_artifacts,
                "orphan_configs": orphan_configs,
                "configs_without_strategy_file": {sid: [str(p.relative_to(root)) for p in paths]
                                                  for sid, paths in config_by_strategy.items() if sid not in entries},
                "entries": entries}
    _save(root / "discovery/library.json", manifest)
    return manifest


def _db(root: Path) -> sqlite3.Connection:
    # ponytail: local single-writer state; coordinate ownership before multi-machine batches.
    path = root / "discovery/state.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS candidates (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, status TEXT NOT NULL, evidence TEXT NOT NULL, attempts TEXT NOT NULL, error TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS creations (seq INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id TEXT UNIQUE NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS batches (start_seq INTEGER PRIMARY KEY, end_seq INTEGER NOT NULL, status TEXT NOT NULL, retry_at REAL, error TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS batch_candidates (start_seq INTEGER NOT NULL, source_id TEXT NOT NULL, candidate_id TEXT NOT NULL, PRIMARY KEY(start_seq, source_id))")
    return db


def record_factory_creation(root: Path, strategy_id: str) -> int | None:
    """Record a persisted factory strategy and queue each complete 200-item range."""
    with _db(root) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM creations WHERE strategy_id=?", (strategy_id,)).fetchone():
            return None
        seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM creations").fetchone()[0]
        db.execute("INSERT INTO creations(seq,strategy_id) VALUES (?,?)", (seq, strategy_id))
        if seq % 200 == 0:
            db.execute("INSERT OR IGNORE INTO batches(start_seq,end_seq,status) VALUES (?,?, 'queued')",
                       (seq - 199, seq))
        return seq


def resume_discovery_batches(root: Path) -> None:
    with _db(root) as db:
        db.execute("UPDATE batches SET status='queued' WHERE status='running'")


class DiscoveryWorker:
    """Process queued factory-creation batches in a daemon thread."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.wake_event = threading.Event()
        resume_discovery_batches(root)

    def run_once(self) -> bool:
        with _db(self.root) as db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute(
                "SELECT start_seq,end_seq FROM batches WHERE status='queued' OR "
                "(status='retry_pending' AND retry_at<=?) ORDER BY "
                "CASE status WHEN 'queued' THEN 0 ELSE 1 END,start_seq LIMIT 1", (time.time(),)
            ).fetchone()
            if batch is None:
                return False
            start_seq, end_seq = batch
            db.execute("UPDATE batches SET status='running',retry_at=NULL,error=NULL WHERE start_seq=?", (start_seq,))
            sources = [row[0] for row in db.execute(
                "SELECT strategy_id FROM creations WHERE seq BETWEEN ? AND ? ORDER BY seq",
                (start_seq, end_seq))]
        try:
            entries = index_library(self.root)["entries"]
            for source_id in sources:
                source = entries.get(source_id)
                if not source or source["wfo"]["status"] != "available":
                    continue
                with _db(self.root) as db:
                    row = db.execute("SELECT candidate_id FROM batch_candidates WHERE start_seq=? AND source_id=?",
                                     (start_seq, source_id)).fetchone()
                if row:
                    candidate = _candidate_by_id(self.root, row[0])
                    if candidate["status"] in {"shortlisted", "rejected", "approved", "review_rejected"}:
                        continue
                    candidate_id = row[0]
                else:
                    for attempt in range(10):
                        try:
                            candidate = generate_candidate(self.root, source_id)
                            break
                        except Exception:
                            if attempt == 9:
                                raise
                    candidate_id = candidate["id"]
                    with _db(self.root) as db:
                        db.execute("INSERT OR IGNORE INTO batch_candidates VALUES (?,?,?)",
                                   (start_seq, source_id, candidate_id))
                evaluate(self.root, candidate_id)
            with _db(self.root) as db:
                db.execute("UPDATE batches SET status='complete',error=NULL WHERE start_seq=?", (start_seq,))
        except Exception as exc:
            with _db(self.root) as db:
                db.execute("UPDATE batches SET status='retry_pending',retry_at=?,error=? WHERE start_seq=?",
                           (time.time() + 3600, str(exc), start_seq))
            log.exception("discovery batch %d-%d failed; retrying in one hour", start_seq, end_seq)
        return True

    def run_forever(self) -> None:
        while True:
            try:
                worked = self.run_once()
            except Exception:
                log.exception("discovery worker iteration failed")
                worked = False
            if not worked:
                self.wake_event.wait(60)
                self.wake_event.clear()


def _candidate_by_id(root: Path, candidate_id: str) -> dict:
    with _db(root) as db:
        return _candidate(db, candidate_id)


def _candidate(db: sqlite3.Connection, sid: str) -> dict:
    row = db.execute("SELECT source_id,status,evidence,attempts,error FROM candidates WHERE id=?", (sid,)).fetchone()
    if not row:
        raise ValueError(f"unknown candidate: {sid}")
    return {"id": sid, "source_id": row[0], "status": row[1], "evidence": json.loads(row[2]),
            "attempts": json.loads(row[3]), "error": row[4]}


def _put(db: sqlite3.Connection, item: dict) -> None:
    db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,evidence=excluded.evidence,attempts=excluded.attempts,error=excluded.error",
               (item["id"], item["source_id"], item["status"], json.dumps(item["evidence"]), json.dumps(item["attempts"]), item.get("error")))
    db.commit()


def _stage(root: Path, sid: str, config: Path, stage: str, group: str, overrides: dict, candidate: bool = False) -> dict:
    runners = {"backtest": run_backtest_stage, "optimize": run_optimize_stage, "wfo": run_wfo_stage}
    result = runners[stage](canonical_config=config, strategy_id=sid,
                            output_runs_dir=root / "discovery/runs" / group,
                            tmp_dir=root / "discovery/tmp" / group, timeout_sec=5400,
                            backtester_root=root, overrides=overrides,
                            strategy_dir=root / "discovery/strategies" if candidate else None)
    return {"parsed": result.parsed, "bundle": str(result.bundle_path),
            "raw_summary": result.raw_summary,
            "summary_sha256": _hash(result.bundle_path / "summary.json")}


def _search_overrides(root: Path) -> dict:
    return {"data": {"symbols": ["SPY"], "timeframe": "1d", "start": SEARCH_START,
                     "end": SEARCH_END, "source": "csv", "root": str(root / "discovery/data/search")},
            "wfo": WINDOWS.copy()}


def backfill(root: Path, ids: list[str]) -> list[dict]:
    manifest = index_library(root)
    out = []
    for sid in ids:
        item = manifest["entries"].get(sid)
        if not item or not item.get("config") or not (yaml.safe_load((root / item["config"]).read_text()) or {}).get("wfo", {}).get("enabled"):
            raise ValueError(f"{sid}: stock WFO config required")
        result = _stage(root, sid, root / item["config"], "wfo", "sources", _search_overrides(root))
        evidence = {"bundle": result["bundle"], "summary": json.loads((Path(result["bundle"]) / "summary.json").read_text()),
                    "summary_sha256": _hash(Path(result["bundle"]) / "summary.json"),
                    "windows_sha256": _hash(Path(result["bundle"]) / "window_results.json"),
                    "config_sha256": item["config_sha256"], "code_sha256": item["code_sha256"],
                    "data_sha256": manifest["search_data"]["sha256"], "costs": item["costs"],
                    "windows": WINDOWS}
        _save(root / "discovery/backfills" / f"{sid}.json", evidence)
        out.append({"id": sid, "n_windows": evidence["summary"].get("n_windows"), "bundle": result["bundle"]})
    index_library(root)
    return out


def generate_candidate(root: Path, source_id: str) -> dict:
    manifest = index_library(root)
    source = manifest["entries"].get(source_id)
    if not source or source["wfo"]["status"] != "available":
        raise ValueError(f"{source_id}: readable WFO evidence required before generation")
    sid = f"disc_{time.time_ns()}"
    if (root / "discovery/strategies" / f"{sid}.py").exists():
        raise ValueError(f"candidate ID collision: {sid}")
    settings = load_settings(root / "factory/config/settings.toml")
    summary = source["wfo"]["summary"]
    windows_path = _bundle(root, source["wfo"]["bundle"]) / "window_results.json"
    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    weak = sorted(({"window": w.get("window_index"), "start": str(w.get("test_start")),
                    "end": str(w.get("test_end")), "sortino": (w.get("test_summary") or {}).get("sortino")}
                   for w in windows if isinstance(w, dict)),
                  key=lambda w: w["sortino"] if isinstance(w["sortino"], (int, float)) else float("inf"))[:3]
    source_analysis = {"oos": summary.get("oos_summary"), "n_windows": summary.get("n_windows"),
                       "parameter_stability": summary.get("parameter_stability"), "weakest_windows": weak}
    prompt = (f"Generate one stock strategy derived from source {source_id}. "
              f"Source idea: {json.dumps((source.get('result') or {}).get('idea'))}. "
              f"Structured WFO analysis: {json.dumps(source_analysis)}. "
              f"Use SPY daily OHLCV only. Explain the source weakness and a genuinely different mechanism. "
              f"Return strict JSON with strategy_id={sid}, one_line_summary, hypothesis, "
              f"novelty_justification, failure_mode, allow_short, complete strategy_file and config_file. "
              f"Strategy must define GeneratedStrategy and one-bar-shifted signals. Config must "
              f"use strategy={sid}, SPY CSV, 2015-01-02 through 2024-12-31, Sortino optimization, "
              f"and WFO enabled. Use only pandas, numpy, dataclasses, typing and backtester contract imports.")
    gen = call_generator(prompt=prompt, provider=settings.generation.provider,
                         cmd=settings.generation.cmd, flags=settings.generation.flags,
                         timeout_sec=settings.generation.generation_timeout_sec)
    p = gen.parsed
    if p["strategy_id"] != sid:
        raise ValueError("generator changed candidate ID")
    validate_static(strategy_id=sid, strategy_src=p["strategy_file"],
                    config_src=p["config_file"], allow_short=p["allow_short"])
    validate_functional(strategy_id=sid, strategy_src=p["strategy_file"],
                        allow_short=p["allow_short"], tmp_dir=root / "discovery/tmp/validate")
    cfg = yaml.safe_load(p["config_file"])
    cfg.update(run_name=sid, strategy=sid, output_root=str(root / "discovery/runs/candidates"))
    cfg["data"] = _search_overrides(root)["data"]
    cfg["wfo"] = WINDOWS.copy()
    source_costs = source["costs"]
    cfg.setdefault("execution", {}).update(commission_bps=source_costs.get("commission_bps", 2),
                                            slippage_bps=source_costs.get("slippage_bps", 5))
    if not cfg.get("optimization", {}).get("param_space"):
        raise ValueError("candidate optimization parameter space is missing")
    cfg["optimization"]["objective"] = "sortino"
    validate_run_config(_from_dict(cfg))
    strategy_path = root / "discovery/strategies" / f"{sid}.py"
    config_path = root / "discovery/configs" / f"{sid}.yaml"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(p["strategy_file"], encoding="utf-8")
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    novelty = {"summary": p["one_line_summary"], "justification": p["novelty_justification"],
               "hypothesis": p["hypothesis"], "failure_mode": p["failure_mode"],
               "source_id": source_id, "source_wfo": source_analysis,
               "source_code_sha256": source["code_sha256"], "source_config_sha256": source["config_sha256"],
               "source_wfo_summary_sha256": source["wfo"]["summary_sha256"],
               "source_wfo_windows_sha256": source["wfo"]["windows_sha256"]}
    _save(root / "discovery/analysis" / f"{sid}.json", novelty)
    item = {"id": sid, "source_id": source_id, "status": "generated",
            "evidence": {"analysis": novelty, "generation_cost_usd": gen.cost_usd,
                         "search_data_sha256": manifest["search_data"]["sha256"],
                         "code_sha256": _hash(strategy_path), "config_sha256": _hash(config_path)},
            "attempts": {}, "error": None}
    with _db(root) as db:
        _put(db, item)
    return item


def _returns(bundle: Path) -> pd.Series:
    frame = pd.read_csv(bundle / "oos_equity_curve.csv", index_col=0, parse_dates=True)
    return frame["equity"].pct_change().dropna()


def _incrementality(source_bundle: Path, candidate_bundle: Path) -> dict:
    try:
        paired = pd.concat([_returns(source_bundle).rename("source"), _returns(candidate_bundle).rename("candidate")], axis=1, join="inner").dropna()
    except (OSError, KeyError, ValueError) as exc:
        return {"status": "missing", "reason": str(exc)}
    if len(paired) < 126:
        return {"status": "missing", "reason": "less than 126 aligned OOS returns"}
    source_equity = (1 + paired["source"]).cumprod()
    combo_equity = (1 + paired.mean(axis=1)).cumprod()
    base, combo = sortino_ratio(source_equity), sortino_ratio(combo_equity)
    corr = paired.corr().loc["source", "candidate"]
    return {"status": "available", "aligned_bars": len(paired), "source_sortino": base,
            "combined_sortino": combo, "correlation": None if pd.isna(corr) else float(corr),
            "passed": math.isfinite(base) and math.isfinite(combo) and combo > base
            and pd.notna(corr) and corr < 0.8}


def evaluate(root: Path, sid: str) -> dict:
    manifest = index_library(root)
    with _db(root) as db:
        item = _candidate(db, sid)
        source = manifest["entries"].get(item["source_id"])
        if not source or source["wfo"]["status"] != "available":
            raise ValueError("source WFO evidence is missing or stale")
        cfg_path = root / "discovery/configs" / f"{sid}.yaml"
        code_path = root / "discovery/strategies" / f"{sid}.py"
        if not cfg_path.is_file() or not code_path.is_file():
            raise ValueError("candidate files are missing")
        if (item["evidence"].get("search_data_sha256") != manifest["search_data"]["sha256"]
                or item["evidence"].get("code_sha256") != _hash(code_path)
                or item["evidence"].get("config_sha256") != _hash(cfg_path)):
            raise ValueError("candidate code, config, or search data changed since generation")
        cfg = yaml.safe_load(cfg_path.read_text())
        analysis = item["evidence"].get("analysis", {})
        summaries = {" ".join((x.get("result") or {}).get("idea", {}).get("one_line_summary", "").split()).casefold()
                     for x in manifest["entries"].values()}
        existing_code_hashes = {x["code_sha256"] for x in manifest["entries"].values()}
        novelty = {"status": "available", "passed": bool(analysis.get("justification"))
                   and " ".join(analysis.get("summary", "").split()).casefold() not in summaries
                   and _hash(code_path) not in existing_code_hashes,
                   "code_sha256": _hash(code_path), "summary": analysis.get("summary")}
        item["evidence"]["novelty"] = novelty
        _put(db, item)
        overrides = _search_overrides(root)
        for stage in ("backtest", "optimize", "wfo"):
            if stage in item["evidence"]:
                continue
            item["attempts"][stage] = item["attempts"].get(stage, 0) + 1
            _put(db, item)
            try:
                item["evidence"][stage] = _stage(root, sid, cfg_path, stage, "candidates", overrides, True)
                item["status"] = stage
                item["error"] = None
            except Exception as exc:
                item["status"], item["error"] = "failed", str(exc)
                _put(db, item)
                raise
            _put(db, item)
        if "stress" not in item["evidence"]:
            costs = cfg.get("execution", {})
            stress = {"execution": {"commission_bps": 2 * costs.get("commission_bps", 2),
                                    "slippage_bps": 2 * costs.get("slippage_bps", 5)}, **overrides}
            item["attempts"]["stress"] = item["attempts"].get("stress", 0) + 1
            _put(db, item)
            try:
                item["evidence"]["stress"] = {**_stage(root, sid, cfg_path, "wfo", "stress", stress, True),
                                                 "costs": stress["execution"]}
            except Exception as exc:
                item["status"], item["error"] = "failed", str(exc)
                _put(db, item)
                raise
            _put(db, item)
        wfo = item["evidence"]["wfo"]["parsed"]
        stressed = item["evidence"]["stress"]["parsed"]
        item["evidence"]["stress"]["passed"] = stressed["oos_sortino"] > 0 and stressed["oos_sortino"] >= 0.5 * wfo["oos_sortino"]
        source_bundle = _bundle(root, source["wfo"]["bundle"])
        item["evidence"]["incrementality"] = _incrementality(source_bundle, Path(item["evidence"]["wfo"]["bundle"]))
        gates = {"novelty": novelty["passed"], "wfo": wfo["n_windows"] >= 8 and wfo["oos_sortino"] > 0 and wfo["oos_n_trades"] > 0,
                 "stress": item["evidence"]["stress"]["passed"],
                 "incrementality": item["evidence"]["incrementality"].get("passed") is True}
        item["evidence"]["gates"] = gates
        item["status"] = "shortlisted" if all(gates.values()) else "rejected"
        item["error"] = None
        _put(db, item)
        return item


def review(root: Path, sid: str) -> dict:
    with _db(root) as db:
        item = _candidate(db, sid)
        if item["status"] != "shortlisted":
            raise ValueError("review requires a shortlisted candidate")
        snap = _snapshot(root, REVIEW_START, REVIEW_END, "review")
        cfg_path = root / "discovery/configs" / f"{sid}.yaml"
        code_path = root / "discovery/strategies" / f"{sid}.py"
        if (item["evidence"].get("code_sha256") != _hash(code_path)
                or item["evidence"].get("config_sha256") != _hash(cfg_path)
                or item["evidence"].get("search_data_sha256") != _hash(root / "discovery/data/search/SPY.csv")):
            raise ValueError("candidate code, config, or search data changed since evaluation")
        cfg = yaml.safe_load(cfg_path.read_text())
        params = item["evidence"]["optimize"]["parsed"]["best_params"]
        overrides = {"strategy_params": params, "data": {"symbols": ["SPY"], "timeframe": "1d",
                     "start": REVIEW_START, "end": REVIEW_END, "source": "csv",
                     "root": str(root / "discovery/data/review")}}
        item["attempts"]["review"] = item["attempts"].get("review", 0) + 1
        _put(db, item)
        try:
            outcome = _stage(root, sid, cfg_path, "backtest", "review", overrides, True)
        except Exception as exc:
            item["error"] = str(exc)
            _put(db, item)
            raise
        item["evidence"]["review"] = {**outcome, "period": [REVIEW_START, REVIEW_END], "data": snap,
                                       "parameters": params, "costs": cfg["execution"],
                                       "code_sha256": _hash(code_path), "config_sha256": _hash(cfg_path)}
        item["status"] = "approved" if outcome["raw_summary"].get("sortino", 0) > 0 and outcome["parsed"]["n_trades"] > 0 else "review_rejected"
        if item["status"] == "approved":
            handoff = {"candidate_id": sid, "source_id": item["source_id"],
                       "review_label": "review data; 2025 was used in existing stock tests",
                       "search_period": [SEARCH_START, SEARCH_END],
                       "search_data_sha256": item["evidence"]["search_data_sha256"],
                       "review": item["evidence"]["review"],
                       "gates": item["evidence"]["gates"], "wfo": item["evidence"]["wfo"],
                       "stress": item["evidence"]["stress"], "incrementality": item["evidence"]["incrementality"]}
            path = root / "discovery/handoffs" / f"{sid}.json"
            _save(path, handoff)
            item["evidence"]["handoff"] = {"path": str(path), "sha256": _hash(path)}
        _put(db, item)
        return item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index")
    sub.add_parser("list")
    p = sub.add_parser("backfill"); p.add_argument("strategy_ids", nargs="+")
    p = sub.add_parser("generate"); p.add_argument("source_id")
    p = sub.add_parser("evaluate"); p.add_argument("candidate_id")
    p = sub.add_parser("review"); p.add_argument("candidate_id")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "index":
        manifest = index_library(root)
        result = {"strategies": len(manifest["entries"]), "result_shards": len(manifest["result_shards"]),
                  "search_data": manifest["search_data"],
                  "wfo_available": sum(x["wfo"]["status"] == "available" for x in manifest["entries"].values()),
                  "wfo_missing": sum(x["wfo"]["status"] == "missing" for x in manifest["entries"].values())}
    elif args.command == "list":
        with _db(root) as db:
            result = [dict(zip(("id", "source_id", "status", "attempts", "error"), row)) for row in db.execute("SELECT id,source_id,status,attempts,error FROM candidates ORDER BY id")]
    elif args.command == "backfill":
        result = backfill(root, args.strategy_ids)
    elif args.command == "generate":
        result = generate_candidate(root, args.source_id)
    elif args.command == "evaluate":
        result = evaluate(root, args.candidate_id)
    else:
        result = review(root, args.candidate_id)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
