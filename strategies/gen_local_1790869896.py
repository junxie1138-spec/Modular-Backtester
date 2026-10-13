from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    anchor_window: int = 20
    rank_window: int = 126
    atr_window: int = 14
    extreme_percentile: float = 0.85
    regime_percentile: float = 0.70
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790869896"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(max(params.anchor_window, params.atr_window + 1) + params.rank_window - 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.anchor_window, params.rank_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not 0.5 < params.extreme_percentile < 1.0:
            raise ValueError("extreme_percentile must be between 0.5 and 1")
        if not 0.5 < params.regime_percentile < 1.0:
            raise ValueError("regime_percentile must be between 0.5 and 1")
        if params.stop_atr <= 0 or params.hold_bars < 1:
            raise ValueError("Stop multiplier and holding period must be positive")
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean().replace(0.0, np.nan)
        volume = data["volume"].clip(lower=0.0)
        volume_sum = volume.rolling(params.anchor_window).sum().replace(0.0, np.nan)
        typical_price = (data["high"] + data["low"] + close) / 3.0
        anchor = (typical_price * volume).rolling(params.anchor_window).sum() / volume_sum
        concentration = volume.pow(2).rolling(params.anchor_window).sum() / volume_sum.pow(2)
        displacement = ((close - anchor) / atr).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "atr": atr,
            "anchor": anchor,
            "concentration": concentration,
            "displacement": displacement,
            "concentration_rank": concentration.rolling(params.rank_window).rank(pct=True),
            "displacement_rank": displacement.rolling(params.rank_window).rank(pct=True),
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        price_rank = indicators["displacement_rank"]
        volume_rank = indicators["concentration_rank"]
        extreme = np.where(price_rank >= params.extreme_percentile, 1,
                           np.where(price_rank <= 1.0 - params.extreme_percentile, -1, 0))
        regime = np.where(volume_rank >= params.regime_percentile, -1,
                          np.where(volume_rank <= 1.0 - params.regime_percentile, 1, 0))
        valid = indicators[["atr", "displacement_rank", "concentration_rank"]].notna().all(axis=1)
        candidates = np.where(valid & (indicators["atr"] > 0.0), extreme * regime, 0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # The prior close's decision fills at this bar's open.
                    stop = opens[i] - side * params.stop_atr * atrs[i - 1]
                stopped = side * (closes[i] - stop) < 0.0
                expired = i - entry_bar + 1 >= params.hold_bars
                if not np.isfinite(closes[i]) or not np.isfinite(stop) or stopped or expired:
                    side = 0
                    stop = np.nan
            elif candidates[i] and np.isfinite(closes[i]) and np.isfinite(atrs[i]):
                side = int(candidates[i])
                entry_bar = i + 1
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    sample = pd.DataFrame({
        "open": 100.0,
        "high": 104.0,
        "low": 96.0,
        "close": [100.0, 97.0, 100.0, 100.0, 103.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        "atr": 2.0,
        "displacement_rank": [0.95, 0.5, 0.5, 0.95, 0.5, 0.5],
        "concentration_rank": [0.1, 0.5, 0.5, 0.9, 0.5, 0.5],
    }, index=index)
    params = StrategyParams(stop_atr=1.0)
    result = GeneratedStrategy.generate_signals(sample, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0]
    quiet = sample.copy()
    quiet["close"] = 100.0
    held = GeneratedStrategy.generate_signals(quiet, derived, None, StrategyParams(hold_bars=2)).data
    assert held["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert (result["size"] > 0.0).all()
