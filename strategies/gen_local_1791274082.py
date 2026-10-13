from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnCorridorParams:
    compression_bars: int = 8
    baseline_bars: int = 32
    compression_ratio: float = 0.55
    cancellation_limit: float = 0.25
    atr_bars: int = 14
    trailing_atr: float = 2.2
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[ReturnCorridorParams]):
    strategy_id = "gen_local_1791274082"

    @classmethod
    def params_type(cls) -> type[ReturnCorridorParams]:
        return ReturnCorridorParams

    def warmup_bars(self, params: ReturnCorridorParams) -> int:
        return int(max(200, params.compression_bars + params.baseline_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: ReturnCorridorParams) -> pd.DataFrame:
        if min(params.compression_bars, params.baseline_bars, params.atr_bars, params.max_hold_bars) < 1:
            raise ValueError("Lookbacks and holding limit must be positive")
        if not 0 < params.compression_ratio < 1 or not 0 <= params.cancellation_limit <= 1:
            raise ValueError("Invalid compression or cancellation threshold")
        if not np.isfinite(params.trailing_atr) or params.trailing_atr <= 0:
            raise ValueError("Trailing ATR multiplier must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        returns = out["return"]
        recent = returns.rolling(params.compression_bars, min_periods=params.compression_bars)
        baseline = returns.shift(params.compression_bars).rolling(params.baseline_bars, min_periods=params.baseline_bars)
        out["corridor_width"] = recent.quantile(0.8) - recent.quantile(0.2)
        out["baseline_width"] = baseline.quantile(0.8) - baseline.quantile(0.2)
        out["return_ceiling"] = recent.max()
        out["net_return"] = recent.sum()
        out["gross_return"] = returns.abs().rolling(params.compression_bars, min_periods=params.compression_bars).sum()
        out["cancellation"] = out["net_return"].abs().div(out["gross_return"].where(out["gross_return"] > 0))
        out["compressed"] = (
            (out["baseline_width"] > 0)
            & (out["corridor_width"] <= params.compression_ratio * out["baseline_width"])
            & (out["cancellation"] <= params.cancellation_limit)
        ).fillna(False)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["entry_ready"] = (
            out["compressed"].shift(1, fill_value=False)
            & (returns > out["return_ceiling"].shift(1))
            & (returns > 0)
            & (close > out["ma200"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReturnCorridorParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        averages = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        active = False
        highest_close = 0.0
        stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            close, atr, average = closes[i], atrs[i], averages[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0 and np.isfinite(average)
            if not valid:
                active = False
                continue
            if active:
                highest_close = max(highest_close, close)
                stop = max(stop, highest_close - params.trailing_atr * atr)
                if close <= stop or close <= average or i - entry_bar >= params.max_hold_bars:
                    active = False
            elif entries[i] and close > average:
                active = True
                highest_close = close
                stop = close - params.trailing_atr * atr
                entry_bar = i
            signals[i] = int(active)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 104.0, 105.0, 103.0, 102.0]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 1.0, 5.0, 5.0],
        "ma200": [90.0] * 5,
        "entry_ready": [True, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, ReturnCorridorParams(trailing_atr=2.0)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0], "Stop must ratchet despite ATR expansion, with exactly one-bar signal delay"
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
