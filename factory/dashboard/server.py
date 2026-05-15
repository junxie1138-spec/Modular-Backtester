from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template

from factory.notify import extract_metric
from factory.results import read_records
from factory.settings_loader import Settings, load_settings

log = logging.getLogger(__name__)


def _enrich(records: list[dict], threshold_metric: str, threshold: float) -> list[dict]:
    """Add `is_good` (WFO threshold cleared) and `is_promoted` (held-out
    promotion passed) flags to each record.
    """
    enriched = []
    for r in records:
        val = extract_metric(r, threshold_metric) if r.get("status") == "complete" else None
        is_good = bool(val is not None and val > threshold)
        promo = r.get("promotion") or {}
        is_promoted = bool(promo.get("passed", False))
        enriched.append({**r, "is_good": is_good, "is_promoted": is_promoted})
    return enriched


def _aggregate(records: list[dict], threshold_metric: str, threshold: float) -> dict[str, Any]:
    total = len(records)
    completes = sum(1 for r in records if r.get("status") == "complete")
    failures_by_stage: dict[str, int] = {}
    above_threshold = 0
    promoted = 0
    promotion_attempted = 0
    screened = 0
    cumulative_spend = 0.0
    for r in records:
        cumulative_spend += float(r.get("generation_cost_usd") or 0.0)
        if r.get("status") == "failed":
            stage = r.get("failed_stage") or "unknown"
            failures_by_stage[stage] = failures_by_stage.get(stage, 0) + 1
        elif r.get("status") == "complete":
            if r.get("screened_out"):
                screened += 1
            val = extract_metric(r, threshold_metric)
            if val is not None and val > threshold:
                above_threshold += 1
            promo = r.get("promotion") or {}
            if promo.get("ran"):
                promotion_attempted += 1
                if promo.get("passed"):
                    promoted += 1
    return {
        "total_cycles": total,
        "completes": completes,
        "failures_by_stage": failures_by_stage,
        "above_threshold": above_threshold,
        "promotion_attempted": promotion_attempted,
        "promoted": promoted,
        "screened": screened,
        "cumulative_spend_usd": cumulative_spend,
        "threshold_metric": threshold_metric,
        "threshold_value": threshold,
    }


def create_app(*, settings: Settings) -> Flask:
    """Build a Flask app bound to one Settings (one results store)."""
    here = Path(__file__).parent.resolve()
    app = Flask(
        __name__,
        template_folder=str(here / "templates"),
        static_folder=str(here / "static"),
    )

    @app.get("/")
    def overview():
        records = read_records(settings.paths.results_store)
        # Newest first for the table.
        records = list(reversed(records))
        enriched = _enrich(
            records,
            threshold_metric=settings.alerts.alert_threshold_metric,
            threshold=settings.alerts.alert_threshold,
        )
        summary = _aggregate(
            records,
            threshold_metric=settings.alerts.alert_threshold_metric,
            threshold=settings.alerts.alert_threshold,
        )
        return render_template(
            "overview.html",
            records=enriched,
            summary=summary,
            auto_refresh_sec=settings.dashboard.auto_refresh_sec,
        )

    @app.get("/api/records")
    def api_records():
        records = read_records(settings.paths.results_store)
        return jsonify(records)

    @app.get("/api/summary")
    def api_summary():
        records = read_records(settings.paths.results_store)
        return jsonify(_aggregate(
            records,
            threshold_metric=settings.alerts.alert_threshold_metric,
            threshold=settings.alerts.alert_threshold,
        ))

    @app.get("/strategy/<sid>")
    def detail(sid: str):
        records = read_records(settings.paths.results_store)
        match = next((r for r in records if r.get("strategy_id") == sid), None)
        if match is None:
            return ("not found", 404)
        return render_template("detail.html", record=match)

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser("dashboard.server")
    parser.add_argument(
        "--settings",
        default="factory/config/settings.toml",
        type=Path,
    )
    args = parser.parse_args(argv)
    s = load_settings(args.settings)
    app = create_app(settings=s)
    app.run(host=s.dashboard.host, port=s.dashboard.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
