from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class NotifyConfig:
    alert_threshold_metric: str  # e.g. "wfo.oos_sharpe"
    alert_threshold: float
    telegram_bot_token: str
    telegram_chat_id: str
    dashboard_base_url: str


@dataclass(slots=True, frozen=True)
class NotifyResult:
    eligible: bool
    sent: bool
    reason: str = ""


def extract_metric(record: dict, dotted_path: str) -> Optional[float]:
    """Walk a dotted path in the record. Returns None for any missing step."""
    cur: Any = record
    for part in dotted_path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def format_alert_message(record: dict, *, dashboard_base_url: str) -> str:
    """Build the Telegram message body.

    Always labels the alert as a SHORTLIST SIGNAL, never as a verdict
    (spec §9 landmine 1 — multiple-comparisons / overfitting risk).
    """
    sid = record["strategy_id"]
    summary = (record.get("idea") or {}).get("one_line_summary", "(no summary)")
    wfo = record.get("wfo") or {}
    parts = [
        "[SHORTLIST SIGNAL — not a verdict]",
        f"Strategy: {sid}",
        f"Idea: {summary}",
        f"OOS Sharpe: {wfo.get('oos_sharpe', 'n/a')}",
        f"OOS total return: {wfo.get('oos_total_return', 'n/a')}",
        f"OOS max drawdown: {wfo.get('oos_max_drawdown', 'n/a')}",
        f"OOS trades: {wfo.get('oos_n_trades', 'n/a')}",
        "",
        "This cleared the configured threshold metric on a single historical",
        "path. A held-out gate (different symbol or fully unseen period) is",
        "required before treating this as a real candidate.",
        "",
        f"Detail: {dashboard_base_url.rstrip('/')}/strategy/{sid}",
    ]
    return "\n".join(parts)


def _post_telegram(*, bot_token: str, chat_id: str, text: str) -> bool:
    """POST to the Telegram Bot API sendMessage endpoint. Returns True on 2xx,
    False on any error (logged but swallowed)."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if 200 <= resp.status < 300:
                return True
            log.warning("telegram non-2xx status: %s", resp.status)
            return False
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("telegram post failed: %s", exc)
        return False


def maybe_send_alert(record: dict, cfg: NotifyConfig) -> NotifyResult:
    """Send a Telegram alert iff the threshold metric clears the threshold.

    Never raises. Telegram errors are logged and returned as sent=False with
    reason='telegram_error'.
    """
    if record.get("status") != "complete":
        return NotifyResult(eligible=False, sent=False, reason="not_complete")
    value = extract_metric(record, cfg.alert_threshold_metric)
    if value is None:
        return NotifyResult(eligible=False, sent=False, reason="metric_missing")
    if value < cfg.alert_threshold:
        return NotifyResult(eligible=False, sent=False, reason="below_threshold")
    if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
        log.info("alert eligible but telegram credentials not configured; skipping")
        return NotifyResult(eligible=True, sent=False, reason="no_credentials")

    text = format_alert_message(record, dashboard_base_url=cfg.dashboard_base_url)
    ok = _post_telegram(
        bot_token=cfg.telegram_bot_token,
        chat_id=cfg.telegram_chat_id,
        text=text,
    )
    if not ok:
        return NotifyResult(eligible=True, sent=False, reason="telegram_error")
    return NotifyResult(eligible=True, sent=True, reason="sent")
