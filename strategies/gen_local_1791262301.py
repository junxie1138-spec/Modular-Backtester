from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    correlation_window: int = 40
    atr_window: int = 14
    entry_score: float = 0.20
    breakeven_pct: float = 0.004
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1791262301"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(params.correlation_window + 5, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if (params.correlation_window < 5 or params.atr_window < 1
                or params.entry_score <= 0 or params.breakeven_pct <= 0
                or params.trail_atr <= 0):
            raise ValueError("Windows and thresholds must be positive; correlation_window >= 5.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close.pct_change(fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["lag1"] = out["acceleration"].shift(1)
        out["lag2"] = out["acceleration"].shift(2)
        window = params.correlation_window
        rolling = out["acceleration"].rolling(window, min_periods=window)
        out["rho1"] = rolling.corr(out["lag1"]).shift(1)
        out["rho2"] = rolling.corr(out["lag2"]).shift(1)
        out["volatility"] = out["roc"].rolling(window, min_periods=window).std()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["expected_acceleration"] = (
            out["rho1"] * out["acceleration"] + out["rho2"] * out["lag1"]
        ) / 2.0
        out["expected_return"] = out["roc"] + out["expected_acceleration"]
        out["score"] = out["expected_return"] / out["volatility"].where(
            out["volatility"] > 0
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["rho1"] < -0.10) & (out["rho2"] > 0.15)
            & (out["acceleration"] < 0) & (out["lag1"] > 0)
            & (out["roc"] <= 0) & (out["score"] > params.entry_score)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveParams,
    ) -> SignalFrame:
        count = len(data)
        target = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.25, dtype=float)
        active = False
        entry = peak = stop = held_size = 0.0
        age = 0
        armed = False
        for i in range(count):
            row = data.iloc[i]
            atr = float(indicators["atr"].iloc[i])
            valid = np.isfinite(row[["open", "high", "low", "close"]].to_numpy(dtype=float)).all()
            valid = valid and row["open"] > 0 and np.isfinite(atr) and atr > 0
            if not valid:
                active = False
                continue
            if not active and i > 0 and target[i - 1] == 1:
                entry = float(row["open"])
                peak = entry
                stop = entry - params.trail_atr * float(indicators["atr"].iloc[i - 1])
                held_size = sizes[i - 1]
                age = 0
                armed = False
                active = True
            if active:
                age += 1
                # Test the previously established stop before updating it.
                exited = float(row["low"]) <= stop
                peak = max(peak, float(row["high"]))
                armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr)
                    # A newly raised stop is confirmed by the close.
                    exited = exited or float(row["close"]) <= stop
                if exited or age >= 2:
                    active = False
                    continue
                target[i] = 1
                sizes[i] = held_size
            elif bool(indicators["entry"].iloc[i]):
                score = float(indicators["score"].iloc[i])
                if np.isfinite(score):
                    target[i] = 1
                    sizes[i] = float(np.clip(
                        0.25 + 0.70 * (score - params.entry_score)
                        / (score + params.entry_score), 0.25, 0.95
                    ))
        df = pd.DataFrame({"signal": target, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5, "high": [100.3] * 5,
        "low": [99.8] * 5, "close": [100.0] * 5, "volume": [1000.0] * 5,
    }, index=index)
    ind = pd.DataFrame({
        "atr": [1.0] * 5, "entry": [True, False, False, False, False],
        "score": [0.6] * 5,
    }, index=index)
    strategy = GeneratedStrategy()
    params = WaveParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result["size"].gt(0).all() and result["size"].iloc[1] > 0.25
    data.loc[index[1], ["high", "close"]] = [102.0, 100.2]
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    flat = strategy.indicators(data.iloc[:1], params)
    assert not flat["entry"].any()


if __name__ == "__main__":
    _self_check()
