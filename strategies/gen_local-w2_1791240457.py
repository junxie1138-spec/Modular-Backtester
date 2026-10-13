from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityParams:
    window: int = 10
    depletion_ratio: float = 0.65
    polarization_threshold: float = 0.35
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VolatilityParams]):
    strategy_id = "gen_local-w2_1791240457"

    @classmethod
    def params_type(cls) -> type[VolatilityParams]:
        return VolatilityParams

    def warmup_bars(self, params: VolatilityParams) -> int:
        return max(200, 2 * params.window + 1)

    def indicators(self, data: pd.DataFrame, params: VolatilityParams) -> pd.DataFrame:
        if params.window < 2 or not 0 < params.depletion_ratio < 1:
            raise ValueError("window must be >= 2 and depletion_ratio must be between 0 and 1")
        if not 0 < params.polarization_threshold < 1:
            raise ValueError("polarization_threshold must be between 0 and 1")
        if params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError("profit_target and max_hold_bars must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["up_vol"] = out["return"].clip(lower=0).rolling(params.window).std(ddof=0)
        out["down_vol"] = out["return"].clip(upper=0).rolling(params.window).std(ddof=0)
        out["prior_up_vol"] = out["up_vol"].shift(params.window)
        out["prior_down_vol"] = out["down_vol"].shift(params.window)
        total_vol = out["up_vol"] + out["down_vol"]
        out["polarization"] = (out["up_vol"] - out["down_vol"]) / total_vol.where(total_vol > 0)
        out["drift"] = out["return"].rolling(params.window).sum()
        out["regime_ma"] = close.rolling(200).mean()
        ready = out.notna().all(axis=1)
        long_entry = (
            ready & (close > out["regime_ma"]) & (out["drift"] > 0)
            & (out["polarization"] > params.polarization_threshold)
            & (out["prior_down_vol"] > 0)
            & (out["down_vol"] < params.depletion_ratio * out["prior_down_vol"])
        )
        short_entry = (
            ready & (close < out["regime_ma"]) & (out["drift"] < 0)
            & (out["polarization"] < -params.polarization_threshold)
            & (out["prior_up_vol"] > 0)
            & (out["up_vol"] < params.depletion_ratio * out["prior_up_vol"])
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolatilityParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if position:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid else np.nan
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if not valid or target_hit or expired:
                    position = 0
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = 0.0
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0,
         "close": [100.0, 100.0, 103.0, 100.0, 100.0, 97.0, 100.0],
         "volume": 1000.0},
        index=index,
    )
    entries = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    expected = [0, 1, 1, 0, -1, -1, 0]
    result = strategy.generate_signals(data, entries, None, VolatilityParams())
    assert result.data["signal"].tolist() == expected
    data["close"] = 100.0
    result = strategy.generate_signals(data, entries, None, VolatilityParams(max_hold_bars=2))
    assert result.data["signal"].tolist() == expected


if __name__ == "__main__":
    _self_check()
