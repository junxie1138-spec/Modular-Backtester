from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WeeklyGapParams:
    correlation_window: int = 30
    percentile_window: int = 80
    entry_percentile: float = 0.8
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[WeeklyGapParams]):
    strategy_id = "gen_local_1791307668"

    @classmethod
    def params_type(cls) -> type[WeeklyGapParams]:
        return WeeklyGapParams

    def warmup_bars(self, params: WeeklyGapParams) -> int:
        return int(max(200, params.correlation_window + params.percentile_window + 5, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WeeklyGapParams) -> pd.DataFrame:
        if params.correlation_window < 3 or params.percentile_window < 2 or params.atr_window < 1:
            raise ValueError("Indicator windows must be positive and sufficiently long")
        if not 0.0 < params.entry_percentile <= 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0 or params.hold_bars < 1:
            raise ValueError("Invalid entry, stop, or holding parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = (data["open"] / out["previous_close"].replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["weekly_gap"] = out["gap"].shift(5)
        out["autocorrelation"] = out["gap"].rolling(params.correlation_window, min_periods=params.correlation_window).corr(out["weekly_gap"]).replace([np.inf, -np.inf], np.nan)
        out["autocorrelation_percentile"] = out["autocorrelation"].rolling(params.percentile_window, min_periods=params.percentile_window).rank(pct=True)
        out["gap_flow"] = out["gap"].rolling(5, min_periods=5).sum()
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([(data["high"] - data["low"]), (data["high"] - out["previous_close"]).abs(), (data["low"] - out["previous_close"]).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["entry"] = ((data["close"] > out["ma200"]) & (out["autocorrelation"] > 0.0) & (out["autocorrelation_percentile"] >= params.entry_percentile) & (out["gap_flow"] > 0.0) & (out["gap"] > 0.0) & (out["atr"] > 0.0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WeeklyGapParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        stop = np.nan
        age = 0
        for i in range(len(data)):
            if pending:
                # The previous close's entry instruction fills at this open.
                stop = opens[i] - params.stop_atr * atr[i - 1]
                active = bool(np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(stop))
                pending = False
                age = 0
            if active:
                age += 1
                # Freeze the stop at the entry open using the decision bar's ATR.
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= params.hold_bars or not np.isfinite(regime[i]) or closes[i] <= regime[i]:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0.0:
                signals[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": [100.0, 110.0] + [100.0] * 10, "close": [100.0, 109.0, 107.0, 105.0] + [100.0] * 8}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "ma200": 90.0, "entry": [True] + [False] * 11}, index=index)
    params = WeeklyGapParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1] + [0] * 8
    data["open"] = 100.0
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
