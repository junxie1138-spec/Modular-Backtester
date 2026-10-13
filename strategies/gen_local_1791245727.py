from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SessionComplexityParams:
    lookback: int = 30
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[SessionComplexityParams]):
    strategy_id = "gen_local_1791245727"

    @classmethod
    def params_type(cls):
        return SessionComplexityParams

    def warmup_bars(self, params):
        return int(params.lookback) + 3

    def indicators(self, data, params):
        if params.lookback < 6 or params.holding_bars < 1:
            raise ValueError("lookback must be >= 6 and holding_bars >= 1")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["overnight"] = opening / close.shift(1) - 1.0
        out["intraday"] = close / opening - 1.0
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        for session in ("overnight", "intraday"):
            x = out[session].shift(2)
            y = out[session].shift(1)
            z = out[session]
            valid = x.notna() & y.notna() & z.notna()
            # Equal returns retain chronological ordering.
            code = ((x > y).astype(int) * 4
                    + (x > z).astype(int) * 2
                    + (y > z).astype(int)).where(valid)
            out[session + "_pattern"] = code
            probability_columns = []
            for pattern in (0, 1, 3, 4, 6, 7):
                name = session + "_p" + str(pattern)
                out[name] = code.eq(pattern).astype(float).where(valid).rolling(
                    params.lookback, min_periods=params.lookback
                ).mean()
                probability_columns.append(name)
            probabilities = out[probability_columns]
            entropy = -(probabilities * np.log(
                probabilities.where(probabilities > 0, 1.0)
            )).sum(axis=1, min_count=6) / np.log(6.0)
            concentration = ((probabilities - 1.0 / 6.0) ** 2).sum(
                axis=1, min_count=6
            ) * 6.0 / 5.0
            out[session + "_entropy"] = entropy
            out[session + "_concentration"] = concentration
            out[session + "_complexity"] = entropy * concentration
            out[session + "_drift"] = out[session].rolling(
                params.lookback, min_periods=params.lookback
            ).sum()
        out["complexity_spread"] = (
            out["intraday_complexity"] - out["overnight_complexity"]
        )
        absorption = (
            (out["complexity_spread"] > 0.03)
            & (out["overnight_entropy"] > 0.7)
            & (out["overnight_drift"] < 0)
            & (out["intraday_drift"] > 0)
        )
        discovery = (
            (out["complexity_spread"] < -0.03)
            & (out["intraday_entropy"] > 0.7)
            & (out["overnight_drift"] > 0)
            & (out["intraday_drift"] < 0)
        )
        out["regime"] = np.select([absorption, discovery], [1, 2], default=0)
        out["entry"] = (
            out["regime"].ne(0)
            & out["regime"].ne(out["regime"].shift(1).fillna(0))
        )
        return out

    def generate_signals(self, data, indicators, ctx, params):
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be >= 1")
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i, enter in enumerate(entries):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    # Force a flat bar; ignore entries on the expiry bar.
                    exit_at = -1
            elif enter:
                signal[i] = 1
                exit_at = i + int(params.holding_bars)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SessionComplexityParams()
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    t = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(0.001 * t + 0.015 * np.sin(t))
    opening = close * np.exp(0.004 * np.cos(t * 1.7))
    data = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) * 1.01,
        "low": np.minimum(opening, close) * 0.99,
        "close": close, "volume": 1000000.0
    }, index=index)
    derived = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(
        derived.iloc[:80], strategy.indicators(data.iloc[:80], params)
    )
    assert not derived["entry"].iloc[:strategy.warmup_bars(params) - 1].any()
    context = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    forced = pd.DataFrame({"entry": True}, index=index)
    result = strategy.generate_signals(data, forced, context, params).data
    n = params.holding_bars
    assert result["signal"].iloc[:n + 2].tolist() == [0] + [1] * n + [0]
    assert result["signal"].iloc[n + 2:2 * n + 2].tolist() == [1] * n
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
