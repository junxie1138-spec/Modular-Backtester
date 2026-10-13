from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    frac_order: float = 0.65
    frac_terms: int = 32
    recent_window: int = 3
    baseline_window: int = 16
    percentile_window: int = 126
    entry_percentile: float = 0.85
    atr_window: int = 14
    stop_atr: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790879536"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(
            params.atr_window + 1,
            params.frac_terms + params.recent_window
            + params.baseline_window + params.percentile_window + 2,
        ))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        p = params
        if not (0.0 < p.frac_order < 1.0 and 0.5 < p.entry_percentile < 1.0):
            raise ValueError("Fractional order and entry percentile are out of range.")
        if min(p.frac_terms, p.recent_window, p.baseline_window,
               p.percentile_window, p.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if p.stop_atr <= 0.0 or p.hold_bars not in (1, 2):
            raise ValueError("Use a positive stop and a one- or two-session hold.")

        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        weights = np.ones(p.frac_terms)
        for lag in range(1, p.frac_terms):
            weights[lag] = weights[lag - 1] * (lag - 1 - p.frac_order) / lag
        # Causal finite fractional filter; each lag operates on the entire series.
        out["frac_close"] = out["log_close"] * weights[0]
        for lag in range(1, p.frac_terms):
            out["frac_close"] += weights[lag] * out["log_close"].shift(lag)
        out["frac_change"] = out["frac_close"].diff()
        out["recent_mean"] = out["frac_close"].rolling(p.recent_window).mean()
        out["baseline_mean"] = (
            out["frac_close"].shift(p.recent_window)
            .rolling(p.baseline_window).mean()
        )
        out["noise"] = (
            out["frac_change"].shift(p.recent_window)
            .rolling(p.baseline_window).std()
        )
        out["score"] = (
            (out["recent_mean"] - out["baseline_mean"])
            / out["noise"].where(out["noise"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["threshold"] = (
            out["score"].rolling(p.percentile_window)
            .quantile(p.entry_percentile).shift(1)
        )
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        # An upward threshold crossing identifies the onset, not a persistent level.
        out["entry"] = (
            out["score"].gt(out["threshold"])
            & out["score"].gt(0)
            & out["score"].shift(1).le(out["threshold"].shift(1))
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        entry_bar = -1
        fixed_stop = np.nan
        pending_entry = False

        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    entry_bar = i
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
            if entry_bar >= 0:
                held_bars = i - entry_bar + 1
                if (not np.isfinite(closes[i])
                        or closes[i] < fixed_stop
                        or held_bars >= params.hold_bars):
                    entry_bar = -1
                    fixed_stop = np.nan
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = 1
                pending_entry = True

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 95.0,
        "close": [100.0, 100.0, 98.0, 100.0, 100.0, 103.0, 102.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0,
        "entry": [False, True, False, False, True, False, False, False],
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    # First trade stops after one session; second expires after two.
    assert signals["signal"].tolist() == [0, 0, 1, 0, 0, 1, 1, 0]
    assert signals["size"].gt(0).all()
    flat = data.copy()
    flat["close"] = 100.0
    assert not strategy.indicators(flat, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
