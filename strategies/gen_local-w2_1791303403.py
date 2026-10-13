from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    correlation_window: int = 40
    entry_z: float = 1.2


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local-w2_1791303403"

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    @staticmethod
    def warmup_bars(params: WaveParams) -> int:
        return max(3, int(params.correlation_window)) + 30

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        window = max(3, int(params.correlation_window))
        out = pd.DataFrame(index=data.index)
        out["ma"] = data["close"].rolling(20, min_periods=20).mean()
        out["dispersion"] = data["close"].rolling(20, min_periods=20).std(ddof=0)
        out["z"] = (data["close"] - out["ma"]) / out["dispersion"].replace(0.0, np.nan)
        out["wave"] = out["z"].diff(5)
        out["echo"] = out["wave"].shift(5)
        out["echo_correlation"] = out["wave"].rolling(window, min_periods=window).corr(out["echo"])
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveParams,
    ) -> SignalFrame:
        threshold = max(0.01, float(params.entry_z))
        z = indicators["z"]
        eligible = (
            z.abs().ge(threshold)
            & indicators["echo_correlation"].le(-0.25)
            & (z * indicators["wave"]).gt(0.0)
        ).fillna(False)
        entries = np.where(eligible.to_numpy(), np.where(z.to_numpy() > 0.0, -1, 1), 0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entered_at = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i - entered_at == 7:
                    position = 0
            elif entry != 0:
                position = int(entry)
                entered_at = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=16)
    sample = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"z": 2.0, "wave": 1.0, "echo_correlation": -0.5}, index=index)
    derived.loc[index[2:7], "z"] = -2.0
    derived.loc[index[2:7], "wave"] = -1.0
    result = GeneratedStrategy.generate_signals(sample, derived, None, WaveParams())
    assert result.data["signal"].tolist() == [0] + [-1] * 7 + [0] + [-1] * 7
