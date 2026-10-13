from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTransitionParams:
    transition_window: int = 40
    percentile_window: int = 126
    entry_quantile: float = 0.85
    atr_window: int = 14
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapTransitionParams]):
    strategy_id = "gen_local-w2_1791285021"

    @classmethod
    def params_type(cls):
        return GapTransitionParams

    def warmup_bars(self, params):
        return int(max(params.transition_window + params.percentile_window + 3, params.atr_window + 1))

    def indicators(self, data, params):
        if min(params.transition_window, params.percentile_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.0 < params.entry_quantile < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("Invalid percentile or stop distance.")
        if params.hold_bars not in (1, 2):
            raise ValueError("Holding period must be one or two sessions.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = (data["open"] / out["prior_close"].where(out["prior_close"] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out["gap_sign"] = np.sign(out["gap"])
        out["prior_sign"] = out["gap_sign"].shift(1)
        valid = out["gap_sign"].notna() & out["prior_sign"].notna()
        for label, direction in (("up", 1), ("down", -1)):
            mask = out["prior_sign"].eq(direction)
            out[label + "_count"] = mask.astype(float).where(valid).rolling(params.transition_window).sum().shift(1)
            out[label + "_sum"] = out["gap_sign"].where(mask, 0.0).where(valid).rolling(params.transition_window).sum().shift(1)
            out[label + "_mean"] = out[label + "_sum"] / out[label + "_count"].replace(0, np.nan)
        out["baseline"] = out["gap_sign"].rolling(params.transition_window).mean().shift(1)
        out["conditional"] = np.where(out["gap_sign"] > 0, out["up_mean"], out["down_mean"])
        out["edge"] = (out["conditional"] - out["baseline"]).where(out["gap_sign"].ne(0))
        out["score"] = (out["edge"] * out["gap"].abs()).replace([np.inf, -np.inf], np.nan)
        out["threshold"] = out["score"].abs().rolling(params.percentile_window).quantile(params.entry_quantile).shift(1)
        out["true_range"] = pd.concat([(data["high"] - data["low"]).abs(), (data["high"] - out["prior_close"]).abs(), (data["low"] - out["prior_close"]).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out

    def generate_signals(self, data, indicators, ctx, params):
        raw = np.zeros(len(data), dtype=int)
        position = 0
        pending = False
        entry = distance = 0.0
        age = 0
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        thresholds = indicators["threshold"].to_numpy(dtype=float)
        for i in range(len(data)):
            if position:
                if pending:
                    # Entry occurs today; freeze the previous decision bar's ATR.
                    entry = opens[i]
                    pending = False
                    age = 0
                age += 1
                invalid = not np.isfinite(entry) or entry <= 0 or not np.isfinite(closes[i]) or closes[i] <= 0
                stopped = position * (closes[i] - entry) < -distance
                if invalid or stopped or age >= params.hold_bars:
                    position = 0
                # An exit always produces a flat target before another entry.
            elif np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(scores[i]) and np.isfinite(thresholds[i]) and np.isfinite(atr[i]) and atr[i] > 0 and abs(scores[i]) > thresholds[i]:
                position = int(np.sign(scores[i]))
                distance = params.stop_atr * atr[i]
                pending = True
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def demo():
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 2.0, "score": 1.0, "threshold": 0.5}, index=index)
    strategy = GeneratedStrategy()
    params = GapTransitionParams(stop_atr=1.0)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    for direction in (1, -1):
        sample = data.copy()
        sample.iloc[1, sample.columns.get_loc("open")] = 110.0
        sample.iloc[1, sample.columns.get_loc("close")] = 110.0 - direction * 3.0
        inputs = ind.copy()
        inputs["score"] = float(direction)
        signals = strategy.generate_signals(sample, inputs, ctx, params).data["signal"]
        assert signals.iloc[:3].tolist() == [0, direction, 0]


if __name__ == "__main__":
    demo()
