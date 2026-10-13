from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ImpactParams:
    lookback: int = 20
    volume_threshold: float = 1.15
    impact_multiplier: float = 1.8
    profit_target: float = 0.007
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[ImpactParams]):
    strategy_id = "gen_local_1790861850"

    @classmethod
    def params_type(cls):
        return ImpactParams

    def warmup_bars(self, params: ImpactParams) -> int:
        return 2 * params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: ImpactParams) -> pd.DataFrame:
        if params.lookback < 3 or params.volume_threshold <= 0:
            raise ValueError("lookback must be >= 3 and volume_threshold positive")
        if params.impact_multiplier <= 1 or params.profit_target <= 0:
            raise ValueError("impact_multiplier must exceed 1 and profit_target be positive")
        if params.max_hold not in (1, 2):
            raise ValueError("max_hold must be 1 or 2")
        out = pd.DataFrame(index=data.index)
        out["body"] = data["close"] / data["open"].where(data["open"] > 0) - 1
        out["volume_baseline"] = data["volume"].where(data["volume"] > 0).rolling(
            params.lookback, min_periods=params.lookback
        ).median().shift(1)
        out["relative_volume"] = data["volume"] / out["volume_baseline"]
        out["impact"] = out["body"].abs() / out["relative_volume"].where(
            out["relative_volume"] > 0
        )
        out["impact_baseline"] = out["impact"].rolling(
            params.lookback, min_periods=params.lookback
        ).median().shift(1)
        out["close_location"] = (data["close"] - data["low"]) / (
            data["high"] - data["low"]
        ).where(data["high"] > data["low"])
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["direction"] = np.where(
            ready & (out["body"] > 0) & (out["close_location"] >= 0.7), 1,
            np.where(ready & (out["body"] < 0) & (out["close_location"] <= 0.3), -1, 0)
        )
        out["shock"] = (
            ready & (out["impact_baseline"] > 0)
            & (out["relative_volume"] >= params.volume_threshold)
            & (out["impact"] >= params.impact_multiplier * out["impact_baseline"])
            & (out["direction"] != 0)
        )
        out["onset"] = out["shock"] & ~out["shock"].shift(1, fill_value=False)
        out["candidate"] = np.where(
            out["onset"].shift(1, fill_value=False)
            & (out["direction"] == out["direction"].shift(1))
            & (out["direction"] != 0)
            & (out["relative_volume"] >= params.volume_threshold)
            & ((data["close"] - data["close"].shift(1)) * out["direction"] > 0),
            out["direction"], 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ImpactParams
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position, entry_bar, entry_price = 0, -1, 0.0
        for i in range(len(data)):
            if position:
                # Yesterday's entry decision fills at today's open.
                if entry_bar < 0:
                    entry_bar, entry_price = i, opens[i]
                if (
                    not np.isfinite(entry_price) or entry_price <= 0
                    or not np.isfinite(closes[i]) or closes[i] <= 0
                ):
                    position = 0
                elif (
                    position * (closes[i] / entry_price - 1) >= params.profit_target
                    or i - entry_bar + 1 >= params.max_hold
                ):
                    # Close-observed exits fill at the next open after the shift.
                    position = 0
            elif candidates[i]:
                position, entry_bar = int(candidates[i]), -1
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = ImpactParams(lookback=5)
    index = pd.date_range("2020-01-01", periods=20)
    body = np.full(20, 0.001)
    body[12:14] = [0.01, 0.002]
    close = 100 * np.cumprod(1 + body)
    opening = np.r_[100.0, close[:-1]]
    volume = np.full(20, 1000.0)
    volume[12:14] = [2000.0, 1500.0]
    data = pd.DataFrame({
        "open": opening, "high": close + 0.01, "low": opening - 0.01,
        "close": close, "volume": volume
    }, index=index)
    indicators = strategy.indicators(data, params)
    assert indicators["candidate"].iloc[12] == 0
    assert indicators["candidate"].iloc[13] == 1
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    for side in (-1, 1):
        candidates = pd.DataFrame({"candidate": np.zeros(20, dtype=int)}, index=index)
        candidates.iloc[13, 0] = side
        flat = data.copy()
        flat.loc[:, ["open", "close"]] = 100.0
        signal = strategy.generate_signals(flat, candidates, ctx, params).data["signal"]
        assert signal.iloc[13:17].tolist() == [0, side, side, 0]
        flat.loc[index[14], "close"] = 100 * (1 + side * 0.02)
        signal = strategy.generate_signals(flat, candidates, ctx, params).data["signal"]
        assert signal.iloc[13:17].tolist() == [0, side, 0, 0]
