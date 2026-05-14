"""One-shot Telegram smoke test.

Reads bot_token + chat_id from settings.toml, posts a single test message
to confirm the credentials are working. Does NOT run a cycle.

USAGE:
    python -m factory.scripts.telegram_smoke --settings factory/config/settings.toml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="factory/config/settings.toml", type=Path)
    args = parser.parse_args(argv)

    from factory.notify import NotifyConfig, format_alert_message, maybe_send_alert
    from factory.settings_loader import load_settings
    s = load_settings(args.settings)
    if not s.alerts.telegram_bot_token or not s.alerts.telegram_chat_id:
        print("ERROR: telegram_bot_token / telegram_chat_id not configured in settings.toml")
        return 2

    fake_record = {
        "strategy_id": "gen_telegram_smoke",
        "status": "complete",
        "idea": {"one_line_summary": "telegram smoke test"},
        "backtest": {"sharpe": 0.5},
        "optimize": {"best_score": 0.7},
        "wfo": {"oos_sharpe": 2.0, "oos_total_return": 0.30,
                "oos_max_drawdown": -0.06, "oos_n_trades": 25},
    }
    cfg = NotifyConfig(
        alert_threshold_metric=s.alerts.alert_threshold_metric,
        alert_threshold=s.alerts.alert_threshold,
        telegram_bot_token=s.alerts.telegram_bot_token,
        telegram_chat_id=s.alerts.telegram_chat_id,
        dashboard_base_url=s.alerts.dashboard_base_url,
    )
    print(format_alert_message(fake_record, dashboard_base_url=s.alerts.dashboard_base_url))
    result = maybe_send_alert(fake_record, cfg)
    print(f"NotifyResult: eligible={result.eligible} sent={result.sent} reason={result.reason}")
    return 0 if result.sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
