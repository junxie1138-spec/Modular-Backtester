from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    signature_bars: int = 12
    short_range_bars: int = 10
    baseline_range_bars: int = 60
    compression_ratio: float = 0.80
    area_threshold: float = 0.10
    profit_target: float = 0.045
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791329219"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.signature_bars + 2,
                       params.short_range_bars + 1,
                       params.baseline_range_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.signature_bars, params.short_range_bars,
               params.baseline_range_bars) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not (0.0 < params.compression_ratio <= 1.0):
            raise ValueError("compression_ratio must be in (0, 1]")
        if not (0.0 < params.area_threshold < 0.5):
            raise ValueError("area_threshold must be in (0, 0.5)")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        length = params.signature_bars
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["log_price"] = np.log(close.where(close > 0.0))
        out["log_return"] = out["log_price"].diff()
        out["path_start"] = out["log_price"].shift(length)
        out["displacement"] = out["log_price"] - out["path_start"]
        out["interior_sum"] = out["log_price"].shift(1).rolling(
            length - 1, min_periods=length - 1
        ).sum() - (length - 1) * out["path_start"]
        out["path_variation"] = out["log_return"].abs().rolling(
            length, min_periods=length
        ).sum()
        # Half of integral(t dx - x dt), with time scaled to [0, 1].
        # Positive area means price gains arrive late in the path.
        out["signature_area"] = 0.5 * (
            (1.0 - 1.0 / length) * out["displacement"]
            - (2.0 / length) * out["interior_sum"]
        )
        out["normalized_area"] = out["signature_area"] / out[
            "path_variation"
        ].where(out["path_variation"] > 1e-12)
        out["previous_area"] = out["normalized_area"].shift(1)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["short_range"] = out["true_range"].rolling(
            params.short_range_bars, min_periods=params.short_range_bars
        ).mean()
        out["baseline_range"] = out["true_range"].rolling(
            params.baseline_range_bars, min_periods=params.baseline_range_bars
        ).mean()
        out["range_ratio"] = out["short_range"] / out[
            "baseline_range"
        ].where(out["baseline_range"] > 0.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (close > out["ma200"])
            & (out["range_ratio"] <= params.compression_ratio)
            & (out["displacement"] > 0.0)
            & (out["normalized_area"] > params.area_threshold)
            & (out["previous_area"] <= params.area_threshold)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.max_hold_bars < 1 or not (
            np.isfinite(params.profit_target) and params.profit_target > 0.0
        ):
            raise ValueError("Holding period and profit target must be positive")

        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if active:
                # The preceding close's decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        raise ValueError("Entry open must be finite and positive")
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i]:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 106.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, False, False]
    }, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams(max_hold_bars=3, profit_target=0.04)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 105.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert (result["size"] > 0.0).all()
    flat = strategy.indicators(data, GeneratedParams())
    assert not flat["entry"].any()


if __name__ == "__main__":
    _self_check()
