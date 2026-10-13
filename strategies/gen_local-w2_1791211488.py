from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExcursionParams:
    volatility_window: int = 6
    channel_window: int = 20
    max_volatility_ratio: float = 0.65
    max_channel_position: float = 0.30
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[ExcursionParams]):
    strategy_id = "gen_local-w2_1791211488"

    @classmethod
    def params_type(cls) -> type[ExcursionParams]:
        return ExcursionParams

    @staticmethod
    def warmup_bars(params: ExcursionParams) -> int:
        return int(max(params.channel_window, params.volatility_window + 1))

    def indicators(self, data: pd.DataFrame, params: ExcursionParams) -> pd.DataFrame:
        if params.volatility_window < 2 or params.channel_window < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five")
        if not np.isfinite(params.max_volatility_ratio) or params.max_volatility_ratio <= 0:
            raise ValueError("max_volatility_ratio must be finite and positive")
        if not 0 <= params.max_channel_position <= 1:
            raise ValueError("max_channel_position must lie between zero and one")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["downside_excursion"] = (data["open"] - data["low"]).clip(lower=0) / out["previous_close"]
        out["upside_excursion"] = (data["high"] - data["open"]).clip(lower=0) / out["previous_close"]
        out["downside_std"] = out["downside_excursion"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        out["upside_std"] = out["upside_excursion"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        out["volatility_ratio"] = out["downside_std"] / out["upside_std"].where(out["upside_std"] > 0)
        out["channel_low"] = data["low"].rolling(params.channel_window, min_periods=params.channel_window).min()
        out["channel_high"] = data["high"].rolling(params.channel_window, min_periods=params.channel_window).max()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["channel_position"] = (data["close"] - out["channel_low"]) / out["channel_width"].where(out["channel_width"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _fixed_hold(entries: np.ndarray, holding_bars: int) -> np.ndarray:
        if not 3 <= holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five")
        signal = np.zeros(len(entries), dtype=int)
        exit_bar = -1
        for i, candidate in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force a flat bar even when another entry condition agrees.
                    exit_bar = -1
            elif candidate:
                signal[i] = 1
                exit_bar = i + holding_bars
        return signal

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ExcursionParams) -> SignalFrame:
        volatility_agrees = indicators["volatility_ratio"].le(params.max_volatility_ratio)
        position_agrees = indicators["channel_position"].between(0, params.max_channel_position)
        entries = (volatility_agrees & position_agrees).fillna(False).to_numpy(dtype=bool)
        df = pd.DataFrame(index=data.index)
        df["signal"] = self._fixed_hold(entries, params.holding_bars)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    for holding_bars in (3, 4, 5):
        raw = GeneratedStrategy._fixed_hold(np.ones(16, dtype=bool), holding_bars)
        filled = pd.Series(raw).shift(1).fillna(0).astype(int)
        assert filled.iloc[0] == 0
        assert filled.iloc[1:holding_bars + 1].eq(1).all()
        assert filled.iloc[holding_bars + 1] == 0
        assert filled.iloc[holding_bars + 2] == 1
    assert not GeneratedStrategy._fixed_hold(np.zeros(16, dtype=bool), 4).any()
