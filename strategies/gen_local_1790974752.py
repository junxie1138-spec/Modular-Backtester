from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeReboundParams:
    correlation_window: int = 40
    rebound_threshold: float = 0.5
    stop_atr: float = 1.5
    atr_window: int = 14
    holding_days: int = 4


class GeneratedStrategy(BaseStrategy[RangeReboundParams]):
    strategy_id = "gen_local_1790974752"

    @classmethod
    def params_type(cls):
        return RangeReboundParams

    def warmup_bars(self, params: RangeReboundParams) -> int:
        return int(max(200, params.correlation_window + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangeReboundParams) -> pd.DataFrame:
        if (params.correlation_window < 3 or params.atr_window < 1
                or not 3 <= params.holding_days <= 5
                or not np.isfinite(params.rebound_threshold)
                or params.rebound_threshold <= 0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid range-rebound parameters")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["normalized_range"] = span / previous_close
        out["range_change"] = out["normalized_range"].diff()
        window = params.correlation_window
        changes = out["range_change"]
        # Fit on prior bars so today's contraction is an out-of-sample shock.
        out["range_autocorr"] = changes.shift(1).rolling(window, min_periods=window).corr(changes.shift(2))
        out["range_scale"] = changes.shift(1).rolling(window, min_periods=window).std(ddof=0)
        out["predicted_rebound"] = out["range_autocorr"] * changes
        out["close_location"] = (data["close"] - data["low"]) / span
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeReboundParams,
    ) -> SignalFrame:
        ready = np.isfinite(indicators[[
            "range_autocorr", "range_scale", "predicted_rebound",
            "close_location", "ma200", "atr",
        ]]).all(axis=1)
        enter = (
            ready
            & (indicators["range_autocorr"] < -0.2)
            & (indicators["range_scale"] > 0)
            & (indicators["predicted_rebound"] > params.rebound_threshold * indicators["range_scale"])
            & (indicators["close_location"] >= 0.65)
            & (data["close"] > data["open"])
            & (data["close"] > indicators["ma200"])
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Entry is today's open; freeze yesterday's known ATR.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                if (not np.isfinite(closes[i]) or not np.isfinite(stop)
                        or closes[i] < stop
                        or i - entry_bar + 1 >= params.holding_days):
                    active = False
            elif enter[i]:
                active = True
                entry_bar = i + 1
            decisions[i] = int(active)
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeReboundParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    for stopped in (False, True):
        data = pd.DataFrame(
            {"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 1000.0},
            index=pd.date_range("2020-01-01", periods=8),
        )
        ind = pd.DataFrame({
            "range_autocorr": [-0.5] + [0.0] * 7,
            "range_scale": 1.0, "predicted_rebound": 1.0,
            "close_location": 0.8, "ma200": 99.0, "atr": 2.0,
        }, index=data.index)
        if stopped:
            data.iloc[2, data.columns.get_loc("close")] = 96.0
            data.iloc[2, data.columns.get_loc("low")] = 95.0
        result = strategy.generate_signals(data, ind, ctx, params).data
        expected = [0, 1, 1, 0, 0, 0, 0, 0] if stopped else [0, 1, 1, 1, 1, 0, 0, 0]
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0).all()
        cold = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
        assert cold["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
