from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpiryImpulseParams:
    volume_window: int = 20
    noise_window: int = 10
    min_volume_ratio: float = 1.2
    min_snr: float = 1.25
    body_threshold: float = 0.25
    profit_target: float = 0.012
    max_hold_bars: int = 4
    min_size: float = 0.35
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ExpiryImpulseParams]):
    strategy_id = "gen_local-w2_1791223262"

    @classmethod
    def params_type(cls):
        return ExpiryImpulseParams

    def warmup_bars(self, params: ExpiryImpulseParams) -> int:
        return int(max(params.volume_window, params.noise_window + 1))

    def indicators(self, data: pd.DataFrame, params: ExpiryImpulseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        daily_return = data["close"].pct_change()
        noise = daily_return.abs().rolling(
            params.noise_window, min_periods=params.noise_window
        ).mean()
        average_volume = data["volume"].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean()
        day_range = (data["high"] - data["low"]).replace(0.0, np.nan)

        out["return"] = daily_return.fillna(0.0)
        out["volume_ratio"] = data["volume"].div(
            average_volume.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["snr"] = daily_return.div(
            noise.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["body_efficiency"] = (data["close"] - data["open"]).div(
            day_range
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["expiry_friday"] = pd.Series(
            (data.index.dayofweek == 4)
            & (data.index.day >= 15)
            & (data.index.day <= 21),
            index=data.index,
            dtype=bool,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ExpiryImpulseParams,
    ) -> SignalFrame:
        entry = (
            indicators["expiry_friday"]
            & indicators["return"].gt(0.0)
            & indicators["volume_ratio"].ge(params.min_volume_ratio)
            & indicators["snr"].ge(params.min_snr)
            & indicators["body_efficiency"].ge(params.body_threshold)
        )

        volume_score = (
            (indicators["volume_ratio"] - params.min_volume_ratio)
            / params.min_volume_ratio
        ).clip(0.0, 1.0)
        snr_score = (
            (indicators["snr"] - params.min_snr) / params.min_snr
        ).clip(0.0, 1.0)
        body_score = indicators["body_efficiency"].clip(0.0, 1.0)
        conviction = (0.4 * volume_score + 0.4 * snr_score + 0.2 * body_score).fillna(0.0)
        scaled_size = params.min_size + conviction * (params.max_size - params.min_size)

        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = entry.to_numpy(dtype=bool)
        candidate_sizes = scaled_size.to_numpy(dtype=float)
        in_position = False
        entry_price = 0.0
        bars_held = 0
        position_size = params.min_size

        for i in range(len(data)):
            if in_position:
                bars_held += 1
                target_hit = closes[i] >= entry_price * (1.0 + params.profit_target)
                timed_out = bars_held >= params.max_hold_bars
                if target_hit or timed_out:
                    in_position = False
                    signals[i] = 0
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif candidates[i]:
                in_position = True
                entry_price = closes[i]
                bars_held = 0
                position_size = candidate_sizes[i]
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(sizes, index=data.index).shift(1).fillna(
            params.min_size
        ).clip(lower=params.min_size, upper=params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2024-01-01", periods=7, freq="B")
    data = pd.DataFrame({"close": [100.0, 100.0, 102.0, 102.0, 102.0, 102.0, 102.0]}, index=index)
    indicators = pd.DataFrame(
        {
            "expiry_friday": [True, False, False, False, False, False, False],
            "return": [0.01] + [0.0] * 6,
            "volume_ratio": [2.0] + [0.0] * 6,
            "snr": [2.0] + [0.0] * 6,
            "body_efficiency": [0.8] + [0.0] * 6,
        },
        index=index,
    )
    result = GeneratedStrategy().generate_signals(data, indicators, None, ExpiryImpulseParams()).data
    assert result["signal"].tolist()[:4] == [0, 1, 1, 0]
    assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
