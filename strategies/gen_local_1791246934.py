from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 8
    hold_bars: int = 4

    def __post_init__(self):
        if not isinstance(self.lookback, int) or self.lookback < 2:
            raise ValueError("lookback must be an integer >= 2")
        if not isinstance(self.hold_bars, int) or not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be 3, 4, or 5")


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791246934"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        open_price = data["open"].where(data["open"] > 0)
        close_price = data["close"].where(data["close"] > 0)
        out["gap_return"] = np.log(open_price / close_price.shift(1))
        out["day_return"] = np.log(close_price / open_price)
        gap_path = pd.Series(0.0, index=data.index)
        day_path = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric level-two signature of the cumulative gap/day path.
        for lag in range(params.lookback - 1, -1, -1):
            gap = out["gap_return"].shift(lag)
            day = out["day_return"].shift(lag)
            area = area + 0.5 * (gap_path * day - day_path * gap)
            gap_path = gap_path + gap
            day_path = day_path + day
        out["gap_net"] = gap_path
        out["day_net"] = day_path
        out["signature_area"] = area
        out["gap_variation"] = out["gap_return"].abs().rolling(params.lookback).sum()
        out["day_variation"] = out["day_return"].abs().rolling(params.lookback).sum()
        denominator = out["gap_variation"] * out["day_variation"]
        out["normalized_area"] = (area / denominator.where(denominator > 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        variation = out["gap_variation"] + out["day_variation"]
        out["efficiency"] = ((gap_path + day_path) / variation.where(variation > 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        out["eligible"] = (
            (out["normalized_area"] > 0.10)
            & (out["efficiency"] > 0.35)
            & (out["gap_net"] > 0)
            & (out["day_net"] > 0)
            & (out["day_return"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        entries = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        exit_at = -1
        for i, enter in enumerate(entries):
            if i < exit_at:
                signal[i] = 1
            elif i == exit_at:
                continue
            elif enter:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=16, freq="B")
    data = pd.DataFrame(index=index)
    always_enter = pd.DataFrame({"eligible": True}, index=index)
    for hold in (3, 4, 5):
        params = SignatureParams(hold_bars=hold)
        result = strategy.generate_signals(data, always_enter, None, params).data
        expected = [0] + [
            int(i % (hold + 1) < hold) for i in range(len(index) - 1)
        ]
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0).all()
