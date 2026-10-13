from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PulseParams:
    ma_window: int = 5
    entry_z: float = 0.6
    max_z: float = 1.8


class GeneratedStrategy(BaseStrategy[PulseParams]):
    strategy_id = "gen_local-w2_1791315482"

    @classmethod
    def params_type(cls) -> type[PulseParams]:
        return PulseParams

    def warmup_bars(self, params: PulseParams) -> int:
        return int(max(200, params.ma_window + 2))

    def indicators(self, data: pd.DataFrame, params: PulseParams) -> pd.DataFrame:
        if params.ma_window < 2 or not 0 < params.entry_z < params.max_z:
            raise ValueError("Require ma_window >= 2 and 0 < entry_z < max_z")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["regime_ma"] = close.rolling(200, min_periods=200).mean()
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["delta_z"] = out["z"].diff()
        out["previous_delta_z"] = out["delta_z"].shift(1)
        out["long_entry"] = (
            (close > out["regime_ma"])
            & out["z"].between(params.entry_z, params.max_z)
            & (out["delta_z"] > 0)
            & (out["previous_delta_z"] <= 0)
        ).fillna(False)
        out["short_entry"] = (
            (close < out["regime_ma"])
            & out["z"].between(-params.max_z, -params.entry_z)
            & (out["delta_z"] < 0)
            & (out["previous_delta_z"] >= 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PulseParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        rows = zip(
            indicators["long_entry"].fillna(False).to_numpy(dtype=bool),
            indicators["short_entry"].fillna(False).to_numpy(dtype=bool),
            indicators["delta_z"].to_numpy(dtype=float),
        )
        for i, (enter_long, enter_short, change) in enumerate(rows):
            # Regime and displacement gate entries; only pulse reversal exits.
            if position == 1:
                if np.isfinite(change) and change <= 0:
                    position = 0
            elif position == -1:
                if np.isfinite(change) and change >= 0:
                    position = 0
            elif enter_long:
                position = 1
            elif enter_short:
                position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame(
        {
            "long_entry": [True, False, False, False, False, False],
            "short_entry": [False, False, True, True, False, False],
            "delta_z": [1.0, 0.5, -0.1, -0.3, 0.1, 0.0],
        },
        index=index,
    )
    result = GeneratedStrategy().generate_signals(data, indicators, None, PulseParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
