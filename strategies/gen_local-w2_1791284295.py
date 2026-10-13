from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    signature_window: int = 12
    channel_window: int = 30
    area_threshold: float = 0.06
    recovery_level: float = 0.35
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791284295"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.signature_window, params.channel_window) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.signature_window < 2 or params.channel_window < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.0 < params.area_threshold <= 0.5:
            raise ValueError("area_threshold must be in (0, 0.5].")
        if not 0.0 < params.recovery_level < 1.0 or params.hold_bars < 1:
            raise ValueError("Invalid recovery level or holding period.")

        out = pd.DataFrame(index=data.index)
        length = params.signature_window
        out["log_price"] = np.log(data["close"].where(data["close"] > 0))
        out["log_volume"] = np.log(data["volume"].where(data["volume"] > 0))
        out["dx"] = out["log_price"].diff()
        out["dy"] = out["log_volume"].diff()
        out["start_x"] = out["log_price"].shift(length)
        out["start_y"] = out["log_volume"].shift(length)
        out["net_x"] = out["log_price"] - out["start_x"]
        out["net_y"] = out["log_volume"] - out["start_y"]
        out["edge_area"] = (
            out["log_price"].shift(1) * out["dy"]
            - out["log_volume"].shift(1) * out["dx"]
        )
        # Antisymmetric second-order signature of the translated price-volume path.
        out["signed_area"] = 0.5 * (
            out["edge_area"].rolling(length, min_periods=length).sum()
            - out["start_x"] * out["net_y"]
            + out["start_y"] * out["net_x"]
        )
        out["price_variation"] = out["dx"].abs().rolling(length, min_periods=length).sum()
        out["volume_variation"] = out["dy"].abs().rolling(length, min_periods=length).sum()
        out["area_scale"] = out["price_variation"] * out["volume_variation"]
        out["signature_area"] = (
            out["signed_area"] / out["area_scale"].where(out["area_scale"] > 1e-12)
        )
        out["channel_low"] = data["low"].rolling(params.channel_window).min()
        out["channel_high"] = data["high"].rolling(params.channel_window).max()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["relative_position"] = (
            (data["close"] - out["channel_low"])
            / out["channel_width"].where(out["channel_width"] > 0)
        )
        out["previous_position"] = out["relative_position"].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        entries = (
            (indicators["signature_area"] < -params.area_threshold)
            & (indicators["previous_position"] < params.recovery_level)
            & (indicators["relative_position"] >= params.recovery_level)
        ).fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    raw[i] = 1
                else:
                    exit_bar = -1
                continue
            if enter:
                raw[i] = 1
                exit_bar = i + params.hold_bars

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame(1.0, index=index, columns=["open", "high", "low", "close", "volume"])
    joint = pd.DataFrame(
        {"signature_area": -0.2, "previous_position": 0.2, "relative_position": 0.4},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, joint, ctx, params).data
    expected = ([0] + ([1] * params.hold_bars + [0]) * 3)[:len(data)]
    assert result["signal"].tolist() == expected
    joint["signature_area"] = 0.2
    assert strategy.generate_signals(data, joint, ctx, params).data["signal"].eq(0).all()
    assert strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data["signal"].eq(0).all()
