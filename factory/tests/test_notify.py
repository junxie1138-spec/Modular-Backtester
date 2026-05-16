from unittest import mock

import pytest

from factory.notify import (
    NotifyConfig,
    NotifyResult,
    extract_metric,
    format_alert_message,
    maybe_send_alert,
)


def _record() -> dict:
    return {
        "strategy_id": "gen_42",
        "status": "complete",
        "idea": {"one_line_summary": "compression breakout test"},
        "backtest": {"sharpe": 0.9},
        "optimize": {"best_score": 1.4},
        "wfo": {"oos_sharpe": 1.25, "oos_sortino": 1.40, "oos_total_return": 0.18,
                "oos_max_drawdown": -0.06, "oos_n_trades": 25},
    }


def test_extract_metric_walks_dotted_path() -> None:
    rec = _record()
    assert extract_metric(rec, "wfo.oos_sharpe") == 1.25
    assert extract_metric(rec, "backtest.sharpe") == 0.9
    assert extract_metric(rec, "optimize.best_score") == 1.4


def test_extract_metric_returns_none_for_missing_path() -> None:
    rec = _record()
    assert extract_metric(rec, "wfo.does_not_exist") is None
    assert extract_metric(rec, "missing.thing") is None
    rec_with_none = {"wfo": None}
    assert extract_metric(rec_with_none, "wfo.oos_sharpe") is None


def test_format_alert_message_labels_as_shortlist_signal() -> None:
    msg = format_alert_message(_record(), dashboard_base_url="http://x.y")
    assert "shortlist signal" in msg.lower()
    assert "gen_42" in msg
    assert "compression breakout test" in msg
    assert "1.4" in msg or "1.40" in msg or "1.400" in msg  # oos_sortino
    assert "1.25" in msg or "1.250" in msg  # oos_sharpe
    assert "http://x.y" in msg
    # MUST NOT use words that imply finality.
    bad = {"validated", "winner", "confirmed edge"}
    for term in bad:
        assert term not in msg.lower(), term


def test_maybe_send_alert_skips_when_metric_below_threshold() -> None:
    cfg = NotifyConfig(
        alert_threshold_metric="wfo.oos_sortino",
        alert_threshold=2.0,  # above this record's 1.40
        telegram_bot_token="t",
        telegram_chat_id="c",
        dashboard_base_url="http://x",
    )
    with mock.patch("factory.notify._post_telegram") as post:
        result = maybe_send_alert(_record(), cfg)
    assert result == NotifyResult(eligible=False, sent=False, reason="below_threshold")
    post.assert_not_called()


def test_maybe_send_alert_skips_when_credentials_missing() -> None:
    cfg = NotifyConfig(
        alert_threshold_metric="wfo.oos_sortino",
        alert_threshold=1.0,
        telegram_bot_token="",
        telegram_chat_id="",
        dashboard_base_url="http://x",
    )
    with mock.patch("factory.notify._post_telegram") as post:
        result = maybe_send_alert(_record(), cfg)
    assert result.eligible is True
    assert result.sent is False
    assert result.reason == "no_credentials"
    post.assert_not_called()


def test_maybe_send_alert_calls_telegram_when_above_threshold() -> None:
    cfg = NotifyConfig(
        alert_threshold_metric="wfo.oos_sortino",
        alert_threshold=1.0,
        telegram_bot_token="t",
        telegram_chat_id="c",
        dashboard_base_url="http://x",
    )
    with mock.patch("factory.notify._post_telegram", return_value=True) as post:
        result = maybe_send_alert(_record(), cfg)
    assert result.eligible is True
    assert result.sent is True
    post.assert_called_once()
    kwargs = post.call_args.kwargs
    assert kwargs["bot_token"] == "t"
    assert kwargs["chat_id"] == "c"
    assert "gen_42" in kwargs["text"]


def test_maybe_send_alert_swallows_telegram_failures() -> None:
    cfg = NotifyConfig(
        alert_threshold_metric="wfo.oos_sortino",
        alert_threshold=1.0,
        telegram_bot_token="t",
        telegram_chat_id="c",
        dashboard_base_url="http://x",
    )
    with mock.patch("factory.notify._post_telegram", return_value=False):
        result = maybe_send_alert(_record(), cfg)
    assert result.eligible is True
    assert result.sent is False
    assert result.reason == "telegram_error"
