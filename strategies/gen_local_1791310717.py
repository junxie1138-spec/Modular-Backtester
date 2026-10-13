from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionRecoveryParams:
    peak_lookback: int = 20
    percentile_lookback: int = 126
    tail_quantile: float = 0.80
    volume_quantile: float = 0.75
    profit_target: float = 0.008
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AuctionRecoveryParams]):
    strategy_id = "gen_local_1791310717"

    @classmethod
    def params_type(cls) -> type[AuctionRecoveryParams]:
        return AuctionRecoveryParams

    @staticmethod
    def warmup_bars(params: AuctionRecoveryParams) -> int:
        return int(params.peak_lookback + params.percentile_lookback + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AuctionRecoveryParams) -> pd.DataFrame:
        if params.peak_lookback < 2 or params.percentile_lookback < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.5 < params.tail_quantile < 1.0:
            raise ValueError("tail_quantile must be between 0.5 and 1")
        if not 0.0 < params.volume_quantile < 1.0:
            raise ValueError("volume_quantile must be between 0 and 1")
        p = params.percentile_lookback
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["high"].rolling(params.peak_lookback).max()
        out["drawdown"] = 1.0 - close / out["peak"].replace(0.0, np.nan)
        out["prior_drawdown"] = out["drawdown"].shift(1)
        out["drawdown_threshold"] = out["drawdown"].shift(2).rolling(p).quantile(params.tail_quantile)
        out["volume_threshold"] = data["volume"].shift(1).rolling(p).quantile(params.volume_quantile)
        out["gap"] = data["open"] / close.shift(1).replace(0.0, np.nan) - 1.0
        out["return"] = close / close.shift(1).replace(0.0, np.nan) - 1.0
        out["close_location"] = (close - data["low"]) / (data["high"] - data["low"]).replace(0.0, np.nan)
        out["location_upper"] = out["close_location"].shift(1).rolling(p).quantile(params.tail_quantile)
        out["location_lower"] = out["close_location"].shift(1).rolling(p).quantile(1.0 - params.tail_quantile)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        qualified = ready & (out["prior_drawdown"] > 0.0) & (out["prior_drawdown"] >= out["drawdown_threshold"]) & (data["volume"] > out["volume_threshold"])
        recovery = qualified & (out["gap"] < 0.0) & (out["return"] > 0.0) & (out["close_location"] > out["location_upper"])
        failed_relief = qualified & (out["gap"] > 0.0) & (out["return"] < 0.0) & (out["close_location"] < out["location_lower"])
        out["candidate"] = np.select([recovery, failed_relief], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AuctionRecoveryParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        if not (np.isfinite(opens).all() and np.isfinite(closes).all() and (opens > 0.0).all() and (closes > 0.0).all()):
            raise ValueError("Opening and closing prices must be finite and positive")
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        side, age, entry = 0, 0, np.nan
        for i in range(len(data)):
            if side:
                # The previous close's decision fills at this bar's open.
                if age == 0:
                    entry = opens[i]
                age += 1
                gain = side * (closes[i] / entry - 1.0)
                # Close-observed exits execute at the following open.
                if gain >= params.profit_target or age >= params.hold_bars:
                    side, age, entry = 0, 0, np.nan
            elif candidates[i]:
                side, age = int(candidates[i]), 0
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    for side in (1, -1):
        data = pd.DataFrame({"open": [100.0] * 7, "close": [100.0, 100.0 + side * 2.0, 100.0, 100.0, 100.0, 100.0, 100.0]}, index=index)
        indicators = pd.DataFrame({"candidate": [side, 0, 0, -side, 0, 0, 0]}, index=index)
        params = AuctionRecoveryParams(profit_target=0.01)
        result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, side, 0, 0, -side, -side, 0]
        params.hold_bars = 1
        indicators["candidate"] = [side, 0, 0, 0, 0, 0, 0]
        result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, side, 0, 0, 0, 0, 0]
        assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
