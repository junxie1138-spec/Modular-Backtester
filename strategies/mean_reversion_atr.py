from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.engine.atr import compute_atr
from backtester.engine.tranche_stop import TSPhase
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MeanReversionAtrParams:
    entry_atr_mult: float = 1.25
    mean_lookback: int = 10
    atr_lookback: int = 20
    time_stop_days: int = 7
    runner_time_stop_days: int = 12
    runner_ceiling_atr_mult: float = 1.25
    runtime_trend_threshold: float = 0.0025
    size: float = 1.0


def _ols_slope(window: np.ndarray) -> float:
    """OLS slope of values on bar index. Used in rolling().apply()."""
    x = np.arange(len(window), dtype=float)
    if len(x) < 2 or np.allclose(window, window[0]):
        return 0.0
    cov = np.cov(x, window, bias=True)[0, 1]
    var = float(np.var(x))
    return cov / var if var > 0 else 0.0


class MeanReversionAtrStrategy(BaseStrategy[MeanReversionAtrParams]):
    """Defense-first swing-trading mean-reversion strategy.

    See docs/superpowers/specs/2026-05-14-mean-reversion-atr-design.md section 1.
    """
    strategy_id = "mean_reversion_atr"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"
    uses_multi_symbol = True
    uses_per_bar = True

    @classmethod
    def params_type(cls):
        return MeanReversionAtrParams

    def warmup_bars(self, params: MeanReversionAtrParams) -> int:
        return max(200, params.mean_lookback, params.atr_lookback) + 1

    def indicators(self, data: pd.DataFrame, params: MeanReversionAtrParams) -> pd.DataFrame:
        mean10 = data["close"].rolling(params.mean_lookback).mean()
        atr20 = compute_atr(data, params.atr_lookback)
        log_close = np.log(data["close"])
        slope_log_200d = log_close.rolling(200).apply(_ols_slope, raw=True)
        out = pd.DataFrame(index=data.index)
        out["mean10"] = mean10
        out["atr20"] = atr20
        out["slope_log_200d"] = slope_log_200d
        out["trend_active"] = (np.expm1(slope_log_200d).abs() > params.runtime_trend_threshold).fillna(False)
        return out

    def signal_for_bar(
        self,
        *,
        symbol: str,
        bar_idx: int,
        data_panel: dict[str, pd.DataFrame],
        indicators_panel: dict[str, pd.DataFrame],
        ctx: StrategyContext,
        params: MeanReversionAtrParams,
    ) -> float:
        data = data_panel[symbol]
        indicators = indicators_panel.get(symbol)
        if indicators is None:
            indicators = self.indicators(data, params)
        if bar_idx >= len(data):
            return 0.0

        close = float(data["close"].iloc[bar_idx])
        mean10 = float(indicators["mean10"].iloc[bar_idx])
        atr20 = float(indicators["atr20"].iloc[bar_idx])
        if pd.isna(mean10) or pd.isna(atr20):
            return 0.0  # warmup

        phase = ctx.position_phase.get(symbol)
        regime = getattr(ctx, "regime", None)
        book_flat = (regime is not None and getattr(regime, "book_flat", False))

        # Entry gate (DISARMED + not book_flat + not trend_active + dip below threshold).
        if phase is TSPhase.DISARMED and not book_flat:
            trend_active = bool(indicators["trend_active"].iloc[bar_idx])
            if not trend_active and close <= mean10 - params.entry_atr_mult * atr20:
                return 1.0

        return 0.0

    def generate_signals(self, data, indicators, ctx, params):
        """Legacy v0.3.0 method (unused for per-bar strategies). Empty signal frame."""
        df = pd.DataFrame({"signal": 0.0, "size": params.size}, index=data.index)
        return SignalFrame(data=df)
