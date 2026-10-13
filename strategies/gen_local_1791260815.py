from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    history_bars: int = 504
    hold_bars: int = 18
    atr_bars: int = 14
    min_samples: int = 40
    edge_threshold: float = 0.005
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = "gen_local_1791260815"

    @classmethod
    def params_type(cls):
        return SeasonalParams

    def warmup_bars(self, params):
        return int(max(200, params.atr_bars + 1,
                       params.history_bars + params.hold_bars + 1))

    def indicators(self, data, params):
        if (params.history_bars < 1 or params.hold_bars < 1
                or params.atr_bars < 1 or params.min_samples < 1
                or params.edge_threshold <= 0 or params.breakeven_pct <= 0
                or params.trail_atr <= 0):
            raise ValueError("Lookbacks, sample counts and thresholds must be positive.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["return"] = close.pct_change(fill_method=None)
        out["phase"] = np.minimum((data.index.day - 1) // 7, 3)
        out["phase_start"] = out["phase"].ne(out["phase"].shift(1))
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(params.atr_bars,
                                        min_periods=params.atr_bars).mean()
        # A return is admitted only one session after its entire outcome matured.
        out["mature_return"] = close.pct_change(
            periods=params.hold_bars, fill_method=None).shift(1)
        out["origin_phase"] = out["phase"].shift(params.hold_bars + 1)
        out["baseline"] = out["mature_return"].rolling(
            params.history_bars, min_periods=params.history_bars).mean()
        forecast = pd.Series(np.nan, index=data.index, dtype=float)
        samples = pd.Series(0.0, index=data.index)
        for phase in range(4):
            valid = out["origin_phase"].eq(phase) & out["mature_return"].notna()
            count = valid.astype(float).rolling(
                params.history_bars, min_periods=params.history_bars).sum()
            total = out["mature_return"].where(valid, 0.0).rolling(
                params.history_bars, min_periods=params.history_bars).sum()
            mean = total.div(count.where(count > 0))
            out[f"phase_{phase}_mean"] = mean
            out[f"phase_{phase}_count"] = count
            forecast = forecast.where(out["phase"].ne(phase), mean)
            samples = samples.where(out["phase"].ne(phase), count)
        out["forecast"] = forecast
        out["samples"] = samples
        out["edge"] = forecast - out["baseline"]
        out["ready"] = (
            out[["ma200", "atr", "forecast", "edge"]].notna().all(axis=1)
            & out["samples"].ge(params.min_samples)
            & out["atr"].gt(0) & close.gt(0)
        )
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        target = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        entry_bar = -1
        stop = -np.inf
        armed = False
        pending_atr = np.nan
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            if position:
                if np.isnan(entry):
                    entry = float(data["open"].iloc[i])
                    if not np.isfinite(entry) or entry <= 0:
                        position = 0
                        continue
                    entry_bar = i
                    stop = position * entry - params.trail_atr * pending_atr
                # Stops are close-observed; the resulting flat target fills next bar.
                directed_close = position * close
                if (not np.isfinite(close) or not np.isfinite(atr)
                        or atr <= 0 or directed_close <= stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    position = 0
                    entry = np.nan
                    armed = False
                    continue
                if position * (close / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed stop only rises: actual short stops only fall.
                    stop = max(stop, position * entry,
                               directed_close - params.trail_atr * atr)
            elif bool(indicators["ready"].iloc[i]) and bool(indicators["phase_start"].iloc[i]):
                forecast = float(indicators["forecast"].iloc[i])
                edge = float(indicators["edge"].iloc[i])
                ma = float(indicators["ma200"].iloc[i])
                if close > ma and forecast > 0 and edge >= params.edge_threshold:
                    position = 1
                elif close < ma and forecast < 0 and edge <= -params.edge_threshold:
                    position = -1
                if position:
                    entry = np.nan
                    armed = False
                    pending_atr = atr
            target[i] = position
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SeasonalParams()
    index = pd.date_range("2020-01-01", periods=5, freq="B")
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for side in (1, -1):
        close = 100.0 + side * np.array([0.0, 0.0, 3.0, 0.0, 0.0])
        data = pd.DataFrame({"open": 100.0, "high": close + 1,
                             "low": close - 1, "close": close,
                             "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({
            "atr": 1.0, "ready": True,
            "phase_start": [True, False, False, False, False],
            "forecast": side * 0.03, "edge": side * 0.02,
            "ma200": 100.0 - side,
        }, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, side, 0]
        assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= params.history_bars + params.hold_bars + 1


if __name__ == "__main__":
    _self_check()
