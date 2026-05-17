from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.analytics.drawdown import max_drawdown
from backtester.analytics.exposure import time_in_market, turnover
from backtester.analytics.trades import extract_round_trips


PERIODS_PER_YEAR: dict[str, int] = {"1d": 252, "1h": 1638}


def periods_per_year(timeframe: str) -> int:
    """Annualisation factor — the number of bars in one year for `timeframe`.

    `1d` -> 252 trading days. `1h` -> 1638 = 252 x 6.5 regular-session hours;
    the 6.5 is an approximation (each session's 7th bar is the half-length
    15:30-16:00 bar) — adequate for v1 and tunable here.
    """
    try:
        return PERIODS_PER_YEAR[timeframe]
    except KeyError:
        raise ValueError(
            f"unknown timeframe {timeframe!r}; known: {sorted(PERIODS_PER_YEAR)}"
        ) from None


def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def annualized_return(equity: pd.Series, timeframe: str = "1d") -> float:
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year(timeframe)
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def annualized_volatility(equity: pd.Series, timeframe: str = "1d") -> float:
    r = _returns(equity)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year(timeframe)))


def sharpe_ratio(equity: pd.Series, rf: float = 0.0, timeframe: str = "1d") -> float:
    r = _returns(equity)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    ppy = periods_per_year(timeframe)
    excess = r - (rf / ppy)
    return float(np.sqrt(ppy) * excess.mean() / r.std(ddof=1))


def sortino_ratio(equity: pd.Series, rf: float = 0.0, timeframe: str = "1d") -> float:
    r = _returns(equity)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return 0.0
    ppy = periods_per_year(timeframe)
    excess = r - (rf / ppy)
    return float(np.sqrt(ppy) * excess.mean() / downside.std(ddof=1))


def compute_summary_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    timeframe: str = "1d",
) -> dict:
    if len(equity_curve) == 0:
        return {
            "total_return": 0.0, "annualized_return": 0.0, "annualized_vol": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
            "n_trades": 0, "n_round_trips": 0, "win_rate": 0.0,
            "avg_round_trip_pnl": 0.0, "time_in_market": 0.0, "turnover": 0.0,
            "final_equity": 0.0,
        }

    eq = equity_curve["equity"]
    rts = extract_round_trips(trades) if len(trades) else pd.DataFrame()
    wins = int((rts["pnl"] > 0).sum()) if len(rts) else 0
    win_rate = (wins / len(rts)) if len(rts) else 0.0
    avg_rt = float(rts["pnl"].mean()) if len(rts) else 0.0

    return {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "annualized_return": annualized_return(eq, timeframe),
        "annualized_vol": annualized_volatility(eq, timeframe),
        "sharpe": sharpe_ratio(eq, timeframe=timeframe),
        "sortino": sortino_ratio(eq, timeframe=timeframe),
        "max_drawdown": max_drawdown(eq),
        "n_trades": int(len(trades)),
        "n_round_trips": int(len(rts)),
        "win_rate": float(win_rate),
        "avg_round_trip_pnl": avg_rt,
        "time_in_market": time_in_market(positions),
        "turnover": turnover(trades, equity_curve),
        "final_equity": float(eq.iloc[-1]),
    }
