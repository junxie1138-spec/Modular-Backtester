from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 12
    hold_bars: int = 8
    area_threshold: float = 0.12
    efficiency_threshold: float = 0.25
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790984290"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        if params.lookback < 2 or params.hold_bars < 1:
            raise ValueError("lookback >= 2 and hold_bars >= 1 required")
        if not 0 < params.area_threshold <= 1:
            raise ValueError("area_threshold must be in (0, 1]")
        if not 0 <= params.efficiency_threshold <= 1:
            raise ValueError("efficiency_threshold must be in [0, 1]")
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("size bounds must be in (0, 1]")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        length = params.lookback
        price_step = np.log(data["close"].where(data["close"] > 0)).diff()
        volume_step = np.log(data["volume"].where(data["volume"] > 0)).diff()
        price_step = price_step.replace([np.inf, -np.inf], np.nan)
        volume_step = volume_step.replace([np.inf, -np.inf], np.nan)
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric level-two signature: sum over earlier/later increment pairs.
        for lag in range(1, length):
            cross = price_step.shift(lag) * volume_step - volume_step.shift(lag) * price_step
            area = area + 0.5 * cross.rolling(length - lag, min_periods=length - lag).sum()
        price_distance = price_step.abs().rolling(length, min_periods=length).sum()
        volume_distance = volume_step.abs().rolling(length, min_periods=length).sum()
        momentum = price_step.rolling(length, min_periods=length).sum()
        denominator = (price_distance * volume_distance).replace(0.0, np.nan)
        lead_score = (-2.0 * area / denominator).clip(-1.0, 1.0)
        efficiency = (momentum.abs() / price_distance.replace(0.0, np.nan)).clip(0.0, 1.0)
        valid = lead_score.notna() & efficiency.notna() & momentum.notna()
        direction = np.sign(momentum.fillna(0.0)).astype(int)
        eligible = (
            valid
            & (lead_score.abs() >= params.area_threshold)
            & (efficiency >= params.efficiency_threshold)
            & (lead_score * direction > 0.0)
        )
        return pd.DataFrame({
            "price_step": price_step.fillna(0.0),
            "volume_step": volume_step.fillna(0.0),
            "signature_area": area.fillna(0.0),
            "price_distance": price_distance.fillna(0.0),
            "volume_distance": volume_distance.fillna(0.0),
            "momentum": momentum.fillna(0.0),
            "lead_score": lead_score.fillna(0.0),
            "efficiency": efficiency.fillna(0.0),
            "direction": direction,
            "strength": lead_score.abs().fillna(0.0),
            "eligible": eligible,
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SignatureParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        directions = indicators["direction"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength"].fillna(0.0).clip(0.0, 1.0).to_numpy()
        side = 0
        exit_bar = -1
        entry_size = params.min_size
        for i in range(len(data)):
            if side != 0:
                if i == exit_bar:
                    side = 0
            elif eligible[i] and directions[i] in (-1, 1):
                side = int(directions[i])
                exit_bar = i + params.hold_bars
                entry_size = float(
                    params.min_size + (params.max_size - params.min_size) * strengths[i]
                )
            signals[i] = side
            if side != 0:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SignatureParams(hold_bars=3)
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "eligible": True,
        "direction": [1] + [-1] * 11,
        "strength": np.linspace(0.2, 0.9, 12),
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0, -1, -1, -1]
    expected_size = params.min_size + (params.max_size - params.min_size) * 0.2
    assert np.allclose(result["size"].iloc[1:4], expected_size)
    assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
