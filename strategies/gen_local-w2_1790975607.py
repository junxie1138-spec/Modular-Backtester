from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    corr_window: int = 40
    shock_z: float = 0.6
    hold_bars: int = 4

    def __post_init__(self):
        if self.corr_window < 3:
            raise ValueError("corr_window must be at least 3")
        if not np.isfinite(self.shock_z) or self.shock_z <= 0:
            raise ValueError("shock_z must be finite and positive")
        if self.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local-w2_1790975607"

    @classmethod
    def params_type(cls):
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return max(200, params.corr_window + 3)

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["range_fraction"] = ((data["high"] - data["low"]) / close).where(
            data["high"] > data["low"]
        )
        out["log_range"] = np.log(out["range_fraction"])
        out["range_change"] = out["log_range"].diff()
        out["lag_one"] = out["range_change"].shift(1)
        out["lag_two"] = out["range_change"].shift(2)
        window = params.corr_window
        out["range_sigma"] = out["range_change"].rolling(window).std()
        out["previous_sigma"] = out["range_sigma"].shift(1)
        out["corr_one"] = out["range_change"].rolling(window).corr(out["lag_one"])
        out["corr_two"] = out["range_change"].rolling(window).corr(out["lag_two"])
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["expansion_direction"] = np.sign(out["midpoint"].diff().shift(1))
        out["ma_200"] = close.rolling(200).mean()
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out["alternating"] = (out["corr_one"] < -0.1) & (out["corr_two"] > 0.1)
        out["pulse"] = (
            (out["range_sigma"] > 0)
            & (out["previous_sigma"] > 0)
            & (out["range_change"] < -params.shock_z * out["range_sigma"])
            & (out["lag_one"] > params.shock_z * out["previous_sigma"])
        )
        out["long_ready"] = (
            out["alternating"] & out["pulse"]
            & (out["expansion_direction"] > 0) & (close > out["ma_200"])
        )
        out["short_ready"] = (
            out["alternating"] & out["pulse"]
            & (out["expansion_direction"] < 0) & (close < out["ma_200"])
        )
        out["entry"] = out["long_ready"].astype(int) - out["short_ready"].astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: EchoParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        entered_at = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i - entered_at == params.hold_bars:
                    position = 0
            elif entry != 0:
                position = int(entry)
                entered_at = i
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": [1, -1] * 6}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = GeneratedStrategy().generate_signals(data, indicators, ctx, EchoParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 1]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
