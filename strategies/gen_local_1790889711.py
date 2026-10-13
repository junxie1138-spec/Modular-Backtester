from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.45
    frac_window: int = 32
    noise_window: int = 5
    percentile_window: int = 63
    entry_quantile: float = 0.90
    quiet_quantile: float = 0.30
    atr_window: int = 14
    trail_atr: float = 0.75
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790889711"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.atr_window + 1,
                       params.frac_window + params.noise_window
                       + params.percentile_window + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        windows = (params.frac_window, params.noise_window,
                   params.percentile_window, params.atr_window)
        if any(not isinstance(w, int) or w < 2 for w in windows):
            raise ValueError("Lookback windows must be integers >= 2.")
        if not (0 < params.frac_order < 1
                and 0 < params.quiet_quantile < params.entry_quantile < 1
                and np.isfinite(params.trail_atr) and params.trail_atr > 0
                and params.max_hold in (1, 2)):
            raise ValueError("Invalid fractional order, quantiles, stop, or holding period.")

        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()

        # Fixed-width causal approximation to (1 - L)**d.
        weights = np.ones(params.frac_window, dtype=float)
        for lag in range(1, params.frac_window):
            weights[lag] = -weights[lag - 1] * (params.frac_order - lag + 1) / lag
        fractional = close.copy()
        for lag in range(1, params.frac_window):
            fractional = fractional + weights[lag] * close.shift(lag)
        out["fractional_close"] = fractional
        out["innovation"] = fractional.diff()
        out["prior_noise"] = out["innovation"].rolling(params.noise_window).std().shift(1)
        out["entry_threshold"] = out["innovation"].rolling(
            params.percentile_window).quantile(params.entry_quantile).shift(1)
        out["noise_threshold"] = out["prior_noise"].rolling(
            params.percentile_window).quantile(params.quiet_quantile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        entry = (
            (indicators["innovation"] > 0)
            & (indicators["innovation"] > indicators["entry_threshold"])
            & (indicators["prior_noise"] <= indicators["noise_threshold"])
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if not (np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0):
                active = False
                continue
            if active:
                # Entry filled today if yesterday first requested a long.
                age += 1
                high_water = max(high_water, closes[i])
                stop = max(stop, high_water - params.trail_atr * atrs[i])
                if closes[i] <= stop or age >= params.max_hold:
                    active = False
                    continue
            elif entry[i]:
                active = True
                age = 0
                # Only closes after the next-bar fill belong to this trade.
                high_water = -np.inf
                stop = -np.inf
            signal[i] = int(active)

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"close": [100.0, 102.0, 100.0, 100.0]}, index=index)
    features = pd.DataFrame({
        "innovation": [2.0, 0.0, 0.0, 0.0],
        "entry_threshold": 1.0, "prior_noise": 0.1,
        "noise_threshold": 0.2, "atr": 1.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    params.max_hold = 1
    assert strategy.generate_signals(data, features, ctx, params).data["signal"].tolist() == [0, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
