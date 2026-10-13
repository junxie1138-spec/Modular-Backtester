from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_bars: int = 3
    correlation_bars: int = 50
    acceleration_threshold: float = 0.65
    minimum_correlation: float = 0.15
    atr_bars: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local-w2_1791272663"

    @classmethod
    def params_type(cls) -> type[AccelerationParams]:
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return int(max(200, params.correlation_bars + 3 * params.roc_bars, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        if min(params.roc_bars, params.correlation_bars, params.atr_bars, params.max_hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if params.correlation_bars < 2 or params.acceleration_threshold <= 0:
            raise ValueError("Correlation requires two observations and a positive threshold")
        if not 0 < params.minimum_correlation < 1:
            raise ValueError("minimum_correlation must lie between zero and one")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError("Exit distances must be positive")
        close = data["close"].where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff(params.roc_bars)
        out["lagged_acceleration"] = out["acceleration"].shift(params.roc_bars)
        window = params.correlation_bars
        out["correlation"] = out["acceleration"].rolling(window, min_periods=window).corr(out["lagged_acceleration"]).clip(-1.0, 1.0)
        out["acceleration_scale"] = out["acceleration"].rolling(window, min_periods=window).std(ddof=0).replace(0.0, np.nan)
        out["forecast"] = out["correlation"] * out["acceleration"] / out["acceleration_scale"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1) & (out["atr"] > 0)
        persistent = out["correlation"] >= params.minimum_correlation
        long_setup = valid & persistent & (close > out["ma200"]) & (out["roc"] < 0) & (out["forecast"] >= params.acceleration_threshold)
        short_setup = valid & persistent & (close < out["ma200"]) & (out["roc"] > 0) & (out["forecast"] <= -params.acceleration_threshold)
        out["setup"] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AccelerationParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        side = 0
        pending = 0
        entry = 0.0
        stop_gain = 0.0
        peak_gain = 0.0
        activated = False
        age = 0
        for i in range(len(data)):
            # A decision on the previous close enters at this bar's open.
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    side = pending
                    entry = opens[i]
                    stop_gain = -params.trail_atr * atrs[i - 1]
                    peak_gain = 0.0
                    activated = False
                    age = 0
                pending = 0
            if side:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                    side = 0
                    continue
                gain = side * (closes[i] - entry)
                # Stops are close-evaluated; exits execute at the next open.
                if gain <= stop_gain or age >= params.max_hold_bars:
                    side = 0
                    continue
                peak_gain = max(peak_gain, gain)
                if gain >= entry * params.breakeven_pct:
                    activated = True
                if activated:
                    # Signed favorable-price units make the stop ratchet upward
                    # for both sides; a short's corresponding price stop falls.
                    stop_gain = max(stop_gain, 0.0, peak_gain - params.trail_atr * atrs[i])
                    if gain <= stop_gain:
                        side = 0
                        continue
                decisions[i] = side
            elif setups[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
                pending = int(setups[i])
                decisions[i] = pending
        df = pd.DataFrame({"signal": decisions, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def demo() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 102.0, 100.0, 100.0, 98.0],
        "high": [101.0, 104.0, 103.0, 101.0, 101.0, 101.0],
        "low": [99.0, 99.0, 99.0, 99.0, 96.0, 97.0],
        "close": [100.0, 103.0, 100.0, 100.0, 97.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({"atr": [1.0] * 6, "setup": [1, 0, 0, -1, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = AccelerationParams(trail_atr=2.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 200
    assert strategy.indicators(data, params)["setup"].eq(0).all()


if __name__ == "__main__":
    demo()
