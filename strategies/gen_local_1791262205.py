from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PullbackParams:
    lookback: int = 20
    entry_z: float = 0.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[PullbackParams]):
    strategy_id = "gen_local_1791262205"

    @classmethod
    def params_type(cls) -> type[PullbackParams]:
        return PullbackParams

    def warmup_bars(self, params: PullbackParams) -> int:
        return params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: PullbackParams) -> pd.DataFrame:
        if params.lookback < 2 or not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("lookback must be >= 2 and entry_z must be finite and positive")
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["ma"] = close.rolling(params.lookback, min_periods=params.lookback).mean()
        out["std"] = close.rolling(params.lookback, min_periods=params.lookback).std(ddof=0)
        out["distance"] = close - out["ma"]
        out["z"] = (out["distance"] / out["std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["ma_step"] = out["ma"].diff()
        out["close_step"] = close.diff()
        out["z_step"] = out["z"].diff()
        out["long_setup"] = (
            (out["z"] >= params.entry_z)
            & (out["ma_step"] > 0)
            & (out["close_step"] < 0)
            & (out["z_step"] < 0)
        ).fillna(False)
        out["short_setup"] = (
            (out["z"] <= -params.entry_z)
            & (out["ma_step"] < 0)
            & (out["close_step"] > 0)
            & (out["z_step"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PullbackParams,
    ) -> SignalFrame:
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")
        long_setup = indicators["long_setup"].fillna(False).astype(bool)
        short_setup = indicators["short_setup"].fillna(False).astype(bool)
        confirmed_long = long_setup & long_setup.shift(1, fill_value=False)
        confirmed_short = short_setup & short_setup.shift(1, fill_value=False)
        entries = np.select([confirmed_long, confirmed_short], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position:
                if i == exit_bar:
                    position = 0
            elif entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame(index=index)
    setups = pd.DataFrame({
        "long_setup": [False, True, True, False, False, False, False, False, False, False],
        "short_setup": [False, False, False, True, True, True, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, setups, None, PullbackParams())
    assert result.data["signal"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    result = GeneratedStrategy().generate_signals(data, setups.iloc[:, ::-1].rename(columns={"short_setup": "long_setup", "long_setup": "short_setup"}), None, PullbackParams())
    assert result.data["signal"].tolist() == [0, 0, 0, -1, -1, -1, -1, 0, 0, 0]
