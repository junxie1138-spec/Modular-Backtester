from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ImpulseRecoilParams:
    lookback: int = 60
    impulse_quantile: float = 0.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ImpulseRecoilParams]):
    strategy_id = "gen_local_1791208891"

    @classmethod
    def params_type(cls) -> type[ImpulseRecoilParams]:
        return ImpulseRecoilParams

    def warmup_bars(self, params: ImpulseRecoilParams) -> int:
        return max(2, int(params.lookback)) + 3

    def indicators(self, data: pd.DataFrame, params: ImpulseRecoilParams) -> pd.DataFrame:
        window = max(2, int(params.lookback))
        quantile = float(np.clip(params.impulse_quantile, 0.01, 0.99))
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        out = pd.DataFrame(index=data.index)
        out["return"] = close.pct_change(fill_method=None)
        out["return_threshold"] = out["return"].rolling(window, min_periods=window).quantile(quantile).shift(1)
        out["volume_threshold"] = volume.rolling(window, min_periods=window).quantile(quantile).shift(1)
        out["quiet_threshold"] = volume.rolling(window, min_periods=window).quantile(0.5).shift(1)
        out["impulse"] = (
            (out["return"] > 0)
            & (out["return"] > out["return_threshold"])
            & (volume > out["volume_threshold"])
            & (close > data["open"])
        ).fillna(False)
        out["impulse_displacement"] = close.shift(1) - close.shift(2)
        out["retained_fraction"] = (
            (close - close.shift(2))
            / out["impulse_displacement"].where(out["impulse_displacement"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            out["impulse"].shift(1, fill_value=False)
            & (out["retained_fraction"] > 0.5)
            & (out["retained_fraction"] < 1.0)
            & (volume < out["quiet_threshold"])
            & (volume < volume.shift(1))
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ImpulseRecoilParams,
    ) -> SignalFrame:
        hold = max(1, int(params.hold_bars))
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for bar, enter in enumerate(entries):
            if remaining > 0:
                remaining -= 1
                signal[bar] = int(remaining > 0)
            elif enter:
                remaining = hold
                signal[bar] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=20, freq="B"))
    for hold in (3, 4, 5):
        params = ImpulseRecoilParams(hold_bars=hold)
        for repeated in (False, True):
            entries = [True] * len(data) if repeated else [True] + [False] * (len(data) - 1)
            indicators = pd.DataFrame({"entry": entries}, index=data.index)
            result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
            expected = [0] + [
                int((bar - 1) % (hold + 1) < hold) if repeated else int(bar <= hold)
                for bar in range(1, len(data))
            ]
            assert result["signal"].tolist() == expected
            assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
