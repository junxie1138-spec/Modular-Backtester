from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    ma_bars: int = 40
    coil_bars: int = 8
    z_threshold: float = 1.2
    max_coil_width: float = 0.8
    escape_margin: float = 0.25
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = "gen_local-w2_1791303912"

    @classmethod
    def params_type(cls):
        return BreakoutParams

    @staticmethod
    def warmup_bars(params: BreakoutParams) -> int:
        return params.ma_bars + params.coil_bars + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        if params.ma_bars < 2 or params.coil_bars < 2 or params.hold_bars < 1:
            raise ValueError("Lookbacks must be at least two and holding period positive.")
        thresholds = np.array([params.z_threshold, params.max_coil_width, params.escape_margin])
        if not np.isfinite(thresholds).all() or (thresholds <= 0).any():
            raise ValueError("Thresholds must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        anchor = close.rolling(params.ma_bars, min_periods=params.ma_bars)
        out["ma"] = anchor.mean().shift(1)
        out["dispersion"] = anchor.std(ddof=0).shift(1).replace(0.0, np.nan)
        out["z"] = ((close - out["ma"]) / out["dispersion"]).replace([np.inf, -np.inf], np.nan)
        coil = out["z"].rolling(params.coil_bars, min_periods=params.coil_bars)
        out["coil_high"] = coil.max().shift(2)
        out["coil_low"] = coil.min().shift(2)
        out["coil_width"] = out["coil_high"] - out["coil_low"]
        out["previous_z"] = out["z"].shift(1)
        out["escape_level"] = out["coil_high"] + params.escape_margin
        out["confirmed"] = (
            out["coil_width"].le(params.max_coil_width)
            & out["z"].ge(params.z_threshold)
            & out["previous_z"].ge(params.z_threshold)
            & out["z"].gt(out["escape_level"])
            & out["previous_z"].gt(out["escape_level"])
        ).fillna(False)
        out["entry"] = out["confirmed"] & ~out["confirmed"].shift(1, fill_value=False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BreakoutParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("Holding period must be positive.")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
            elif entry:
                exit_bar = i + params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = BreakoutParams()
    index = pd.date_range("2020-01-01", periods=50, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    flat = GeneratedStrategy.indicators(data, params)
    assert not flat["entry"].any()
    entries = pd.DataFrame({"entry": False}, index=index)
    entries.iloc[1, 0] = True
    entries.iloc[3, 0] = True
    entries.iloc[19, 0] = True
    entries.iloc[24, 0] = True
    result = GeneratedStrategy.generate_signals(data, entries, None, params).data
    assert result["signal"].iloc[:2].eq(0).all()
    assert result["signal"].iloc[2:20].eq(1).all()
    assert result["signal"].iloc[20:25].eq(0).all()
    assert result["signal"].iloc[25:43].eq(1).all()
    assert result["signal"].iloc[43:].eq(0).all()
    assert result["size"].gt(0).all()
