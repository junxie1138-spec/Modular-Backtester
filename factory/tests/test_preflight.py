import json
from pathlib import Path
from types import SimpleNamespace

from factory.scripts.preflight import FAIL, PASS, WARN, _check_hourly_dataset


def _settings(root: Path) -> SimpleNamespace:
    return SimpleNamespace(paths=SimpleNamespace(backtester_root=root))


def _write_report(root: Path, symbols: dict) -> Path:
    report_dir = root / "data" / "raw_hourly"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "_build_report.json"
    path.write_text(json.dumps({"min_hourly_bars": 7000, "symbols": symbols}),
                    encoding="utf-8")
    return path


def test_check_hourly_dataset_warns_when_no_report(tmp_path: Path) -> None:
    status, detail = _check_hourly_dataset(_settings(tmp_path))
    assert status == WARN
    assert "no hourly build report" in detail


def test_check_hourly_dataset_passes_when_spy_tradable(tmp_path: Path) -> None:
    _write_report(tmp_path, {"SPY": {"classification": "tradable",
                                     "bar_count": 18000, "source": "stitched"}})
    status, detail = _check_hourly_dataset(_settings(tmp_path))
    assert status == PASS
    assert "tradable" in detail


def test_check_hourly_dataset_fails_when_spy_insufficient(tmp_path: Path) -> None:
    _write_report(tmp_path, {"SPY": {"classification": "insufficient_history",
                                     "bar_count": 5000, "source": "yfinance_only"}})
    status, detail = _check_hourly_dataset(_settings(tmp_path))
    assert status == FAIL
    assert "insufficient_history" in detail


def test_check_hourly_dataset_fails_when_spy_missing(tmp_path: Path) -> None:
    _write_report(tmp_path, {"AAPL": {"classification": "tradable",
                                      "bar_count": 18000, "source": "stitched"}})
    status, detail = _check_hourly_dataset(_settings(tmp_path))
    assert status == FAIL
    assert "no SPY entry" in detail
