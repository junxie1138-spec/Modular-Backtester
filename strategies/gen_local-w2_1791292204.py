from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_bars: int = 30
    percentile_bars: int = 100
    reset_quantile: float = 0.35
    entry_quantile: float = 0.75
    setup_bars: int = 5
    atr_bars: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold_bars: int = 10

    def __post_init__(self):
        if min(self.ma_bars, self.percentile_bars, self.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0 < self.reset_quantile < self.entry_quantile < 1:
            raise ValueError("Quantiles must be ordered inside (0, 1).")
        if min(self.setup_bars, self.max_hold_bars) < 1:
            raise ValueError("Holding and setup windows must be positive.")
        if not self.breakeven_pct > 0 or not self.trail_atr > 0:
            raise ValueError("Stop parameters must be positive.")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791292204"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.ma_bars + params.percentile_bars + 1,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_bars).mean()
        out["std"] = close.rolling(params.ma_bars).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan)
        history = out["z"].shift(1).rolling(params.percentile_bars)
        out["lower"] = history.quantile(params.reset_quantile)
        out["upper"] = history.quantile(params.entry_quantile)
        out["trend"] = (close > out["ma"]) & (out["ma"] > out["ma"].shift(1))
        previous_close = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_bars).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        values = indicators[["z", "lower", "upper", "atr"]].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        trends = indicators["trend"].fillna(False).to_numpy(dtype=bool)
        active = pending = protected = False
        armed_at = -1
        entry = peak = stop = 0.0
        age = 0

        for i in range(len(data)):
            z, lower, upper, atr = values[i]
            close = closes[i]
            if pending:
                pending = False
                entry = opens[i]
                if not np.isfinite(entry) or entry <= 0:
                    armed_at = -1
                    continue
                active = True
                protected = False
                age = 0
                peak = entry
                stop = entry - params.trail_atr * values[i - 1, 3]

            if active:
                age += 1
                if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                    active = False
                    armed_at = -1
                    continue
                # Stops are close-confirmed; every exit fills at the next open.
                if close <= stop or age >= params.max_hold_bars:
                    active = False
                    armed_at = -1
                    continue
                peak = max(peak, close)
                protected = protected or close >= entry * (1 + params.breakeven_pct)
                if protected:
                    stop = max(stop, entry, peak - params.trail_atr * atr)
                decisions[i] = 1
                continue

            valid = (np.isfinite(values[i]).all() and np.isfinite(close)
                     and close > 0 and atr > 0)
            if not valid or not trends[i] or z <= 0:
                armed_at = -1
                continue
            if armed_at >= 0 and i - armed_at > params.setup_bars:
                armed_at = -1
            if z <= lower:
                armed_at = i
            elif armed_at >= 0 and z >= upper and i > 0 and z > values[i - 1, 0]:
                decisions[i] = 1
                pending = True
                armed_at = -1

        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "close": [100, 100, 102, 104, 102, 100, 100]},
                        index=index)
    indicators = pd.DataFrame({
        "z": [0.5, 3, 3, 3, 3, 3, 3], "lower": 1.0, "upper": 2.0,
        "atr": 1.0, "trend": True
    }, index=index)
    result = GeneratedStrategy().generate_signals(
        data, indicators, None, StrategyParams(trail_atr=1.0)).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
