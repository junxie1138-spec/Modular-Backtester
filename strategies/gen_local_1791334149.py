from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeShockParams:
    baseline_bars: int = 12
    expansion_ratio: float = 1.35
    flow_bars: int = 3
    flow_threshold: float = 0.20
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeShockParams]):
    strategy_id = "gen_local_1791334149"

    @classmethod
    def params_type(cls):
        return RangeShockParams

    def warmup_bars(self, params: RangeShockParams) -> int:
        return max(params.baseline_bars + 1, params.flow_bars + 1, 2)

    def indicators(self, data: pd.DataFrame, params: RangeShockParams) -> pd.DataFrame:
        if (params.baseline_bars < 1 or params.flow_bars < 1
                or params.hold_bars not in (1, 2)
                or not np.isfinite(params.expansion_ratio)
                or params.expansion_ratio <= 1
                or not np.isfinite(params.flow_threshold)
                or not 0 <= params.flow_threshold < 1):
            raise ValueError("Invalid range-shock parameters")
        out = pd.DataFrame(index=data.index)
        out["width"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["relative_width"] = out["width"] / data["close"].where(data["close"] > 0)
        out["baseline"] = out["relative_width"].shift(1).rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean()
        out["expansion"] = out["relative_width"] / out["relative_width"].shift(1)
        out["high_advance"] = data["high"].diff()
        out["low_move"] = data["low"].diff()
        out["volume"] = data["volume"].where(data["volume"] >= 0)
        out["signed_volume"] = np.sign(data["close"].diff()) * out["volume"]
        out["flow_volume"] = out["volume"].rolling(
            params.flow_bars, min_periods=params.flow_bars
        ).sum()
        out["flow_impulse"] = out["signed_volume"].rolling(
            params.flow_bars, min_periods=params.flow_bars
        ).sum() / out["flow_volume"].where(out["flow_volume"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["range_shock"] = (
            (out["relative_width"].shift(1) < out["baseline"])
            & (out["expansion"] >= params.expansion_ratio)
            & (out["high_advance"] > out["low_move"].abs())
        ).fillna(False)
        out["buying_flow"] = (out["flow_impulse"] > params.flow_threshold).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeShockParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        entries = (
            indicators["range_shock"].fillna(False)
            & indicators["buying_flow"].fillna(False)
        ).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                    # The scheduled exit consumes this bar, even with a new setup.
            elif entry:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    gates = pd.DataFrame({"range_shock": True, "buying_flow": True}, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for hold, expected in ((1, [0, 1, 0, 1, 0, 1, 0]),
                           (2, [0, 1, 1, 0, 1, 1, 0])):
        result = strategy.generate_signals(data, gates, ctx, RangeShockParams(hold_bars=hold))
        assert result.data["signal"].tolist() == expected
    gates["buying_flow"] = False
    assert not strategy.generate_signals(data, gates, ctx, RangeShockParams()).data["signal"].any()


if __name__ == "__main__":
    _self_check()
