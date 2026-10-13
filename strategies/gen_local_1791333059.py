from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RotationParams:
    lookback: int = 32
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RotationParams]):
    strategy_id = "gen_local_1791333059"

    @classmethod
    def params_type(cls) -> type[RotationParams]:
        return RotationParams

    def warmup_bars(self, params: RotationParams) -> int:
        return max(3, int(params.lookback)) + 22

    def indicators(self, data: pd.DataFrame, params: RotationParams) -> pd.DataFrame:
        n = max(3, int(params.lookback))
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["clock"] = np.arange(len(data), dtype=float)
        out["slope"] = out["log_close"].rolling(n, min_periods=n).cov(out["clock"], ddof=0) / ((n * n - 1) / 12.0)
        out["innovation"] = out["slope"].diff()
        out["innovation_mean"] = out["innovation"].rolling(20, min_periods=20).mean().shift(1)
        out["innovation_std"] = out["innovation"].rolling(20, min_periods=20).std(ddof=0).shift(1)
        out["range_low"] = data["low"].rolling(n, min_periods=n).min()
        out["range_high"] = data["high"].rolling(n, min_periods=n).max()
        out["range_width"] = out["range_high"] - out["range_low"]
        out["position"] = (data["close"] - out["range_low"]) / out["range_width"].where(out["range_width"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RotationParams) -> SignalFrame:
        required = ["slope", "innovation", "innovation_mean", "innovation_std", "position"]
        ready = np.isfinite(indicators[required]).all(axis=1)
        setup = (
            ready
            & (indicators["slope"].shift(1) <= 0)
            & (indicators["slope"] > 0)
            & (indicators["innovation"] > indicators["innovation_mean"] + 0.5 * indicators["innovation_std"])
            & indicators["position"].between(0.35, 0.65)
        ).fillna(False)
        hold = max(1, int(params.hold_bars))
        signal = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for i, enter in enumerate(setup.to_numpy(dtype=bool)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
            elif enter:
                signal[i] = 1
                exit_bar = i + hold
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "slope": [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
        "innovation": 2.0,
        "innovation_mean": 0.0,
        "innovation_std": 1.0,
        "position": 0.5,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, RotationParams(hold_bars=2))
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 1, 1]
    assert (result.data["size"] > 0).all()
