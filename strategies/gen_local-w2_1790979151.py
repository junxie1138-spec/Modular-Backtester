from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DownsideMigrationParams:
    fast_window: int = 4
    baseline_window: int = 20
    rank_window: int = 60
    entry_percentile: float = 0.85
    reset_percentile: float = 0.45
    atr_window: int = 14
    initial_atr: float = 1.5
    breakeven_pct: float = 0.006
    trail_atr: float = 1.5
    max_hold: int = 4

    def __post_init__(self):
        if not (2 <= self.fast_window <= self.baseline_window
                and self.rank_window >= 2 and self.atr_window >= 2):
            raise ValueError("Invalid lookback windows")
        if not (0 < self.reset_percentile < self.entry_percentile <= 1):
            raise ValueError("Invalid percentile thresholds")
        if not (self.initial_atr > 0 and self.trail_atr > 0
                and self.breakeven_pct > 0 and 3 <= self.max_hold <= 5):
            raise ValueError("Invalid exit parameters")


class GeneratedStrategy(BaseStrategy[DownsideMigrationParams]):
    strategy_id = "gen_local-w2_1790979151"

    @classmethod
    def params_type(cls):
        return DownsideMigrationParams

    def warmup_bars(self, params):
        return int(max(params.baseline_window + params.rank_window + 1,
                       params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["energy"] = out["return"].pow(2)
        out["downside_energy"] = out["return"].clip(upper=0).pow(2)
        for name, window in (("fast", params.fast_window),
                             ("baseline", params.baseline_window)):
            out[name + "_energy"] = out["energy"].rolling(window).sum()
            out[name + "_downside"] = out["downside_energy"].rolling(window).sum()
            out[name + "_share"] = (
                out[name + "_downside"]
                / out[name + "_energy"].replace(0, np.nan)
            )
        out["migration"] = out["baseline_share"] - out["fast_share"]
        out["percentile"] = out["migration"].rolling(
            params.rank_window
        ).rank(method="average", pct=True)
        out["drift"] = out["return"].rolling(params.fast_window).sum()
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["qualifies"] = (
            out["percentile"].ge(params.entry_percentile)
            & out["migration"].gt(0)
            & out["drift"].gt(0)
            & out["atr"].gt(0)
        ).fillna(False)
        out["confirmed"] = (
            out["qualifies"] & out["qualifies"].shift(1, fill_value=False)
        )
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        ranks = indicators["percentile"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = True
        entry = np.nan
        stop = np.nan
        peak = np.nan
        entry_atr = np.nan
        age = 0
        breakeven = False

        for i in range(len(data)):
            open_price, high, low, close = prices[i]
            valid = bool(np.isfinite(prices[i]).all()
                         and (prices[i] > 0).all()
                         and np.isfinite(atrs[i]) and atrs[i] > 0)
            if active:
                # A decision on the previous close fills at this bar's open.
                if np.isnan(entry):
                    entry = open_price
                    stop = entry - params.initial_atr * entry_atr
                    peak = entry
                age += 1
                # Stop breaches are close decisions, exiting at the next open.
                if not valid or low <= stop or age >= params.max_hold:
                    active = False
                    entry = np.nan
                    continue
                peak = max(peak, high)
                if high >= entry * (1.0 + params.breakeven_pct):
                    breakeven = True
                if breakeven:
                    # Updated stops apply to subsequent bars.
                    stop = max(stop, entry, peak - params.trail_atr * atrs[i])
                decisions[i] = 1
            else:
                if np.isfinite(ranks[i]) and ranks[i] <= params.reset_percentile:
                    armed = True
                if armed and valid and confirmed[i]:
                    active = True
                    armed = False
                    entry = np.nan
                    entry_atr = atrs[i]
                    age = 0
                    breakeven = False
                    decisions[i] = 1

        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = DownsideMigrationParams()
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0,
        "high": [100.2, 100.2, 101.0, 100.4, 100.4, 100.4, 100.4, 100.4],
        "low": [99.9, 99.9, 99.9, 99.8, 99.9, 99.9, 99.9, 99.9],
        "close": 100.1,
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0,
        "percentile": 0.95,
        "confirmed": [False, True, True, True, True, True, True, True],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    quiet = data.assign(high=100.2, low=99.9)
    result = strategy.generate_signals(quiet, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    history = pd.concat([quiet] * 20, ignore_index=True)
    history.index = pd.date_range("2020-01-01", periods=len(history))
    derived = strategy.indicators(history, params)
    assert not derived["confirmed"].iloc[:strategy.warmup_bars(params) - 2].any()


if __name__ == "__main__":
    _self_check()
