from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    lookback: int = 6
    stretch: float = 1.2
    recoil_ratio: float = 2.0
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = "gen_local_1790854180"

    @classmethod
    def params_type(cls) -> type[RecoilParams]:
        return RecoilParams

    def warmup_bars(self, params: RecoilParams) -> int:
        if not 3 <= params.lookback <= 9:
            raise ValueError("lookback must be between 3 and 9")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not np.isfinite(params.stretch) or params.stretch <= 0:
            raise ValueError("stretch must be finite and positive")
        if not np.isfinite(params.recoil_ratio) or params.recoil_ratio <= 0:
            raise ValueError("recoil_ratio must be finite and positive")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float)
        window = close.rolling(params.lookback, min_periods=params.lookback)
        mean = window.mean()
        scale = window.std(ddof=0).replace(0.0, np.nan)
        z = ((close - mean) / scale).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["mean"] = mean
        out["scale"] = scale
        out["z"] = z
        out["previous_z"] = z.shift(1)
        out["recoil"] = close.diff()
        out["anchor_retreat"] = (mean.shift(1) - mean).clip(lower=0.0)
        out["ready"] = np.isfinite(out[["z", "previous_z", "recoil", "anchor_retreat"]]).all(axis=1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoilParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entry = (
            indicators["ready"]
            & (indicators["previous_z"] <= -params.stretch)
            & (indicators["z"] < 0.0)
            & (indicators["z"] > indicators["previous_z"])
            & (indicators["recoil"] > params.recoil_ratio * indicators["anchor_retreat"])
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, ready in enumerate(entry):
            if i < exit_at:
                signal[i] = 1
            elif i == exit_at:
                # Emit the scheduled exit even if another entry qualifies.
                continue
            elif ready:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    setups = pd.DataFrame({"ready": True, "previous_z": -2.0, "z": -1.0, "recoil": 1.0, "anchor_retreat": 0.1}, index=index)
    for hold, expected in ((1, [0, 1, 0, 1, 0, 1, 0, 1]), (2, [0, 1, 1, 0, 1, 1, 0, 1])):
        params = RecoilParams(hold_bars=hold)
        result = strategy.generate_signals(data, setups, None, params).data
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0.0).all()
    for lookback in (4, 6, 8):
        params = RecoilParams(lookback=lookback)
        flat = strategy.indicators(data, params)
        assert strategy.warmup_bars(params) <= 10
        assert not flat["ready"].any()
        assert not strategy.generate_signals(data, flat, None, params).data["signal"].any()
