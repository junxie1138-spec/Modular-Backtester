from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReclaimParams:
    volume_window: int = 20
    min_gap: float = 0.0025
    min_demand: float = 0.8
    profit_target: float = 0.02
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GapReclaimParams]):
    strategy_id = "gen_local_1791306159"

    @classmethod
    def params_type(cls):
        return GapReclaimParams

    def warmup_bars(self, params: GapReclaimParams) -> int:
        return max(2, params.volume_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapReclaimParams) -> pd.DataFrame:
        if params.volume_window < 1 or params.max_hold < 1:
            raise ValueError("Lookback and holding period must be positive.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (params.min_gap, params.min_demand, params.profit_target)):
            raise ValueError("Entry thresholds and profit target must be positive and finite.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["gap"] = data["open"].div(out["previous_close"].where(out["previous_close"] > 0)).sub(1)
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["volume_base"] = data["volume"].shift(1).rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean()
        out["relative_volume"] = data["volume"].div(out["volume_base"].where(out["volume_base"] > 0))
        out["body_fraction"] = (data["close"] - data["open"]).div(out["range"])
        out["demand"] = (out["relative_volume"] * out["body_fraction"]).replace(
            [np.inf, -np.inf], np.nan
        )
        out["gap_reclaimed"] = (
            (out["gap"] <= -params.min_gap)
            & (data["open"] < out["previous_low"])
            & (data["close"] > out["previous_close"])
        ).fillna(False)
        out["volume_confirmed"] = (
            (out["relative_volume"] > 1.0)
            & (out["body_fraction"] > 0)
            & (out["demand"] >= params.min_demand)
        ).fillna(False)
        out["entry"] = (out["gap_reclaimed"] & out["volume_confirmed"]).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapReclaimParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Learn the fill price only when the next session actually arrives.
                if i == entry_bar:
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                # Closing-price exits are executed at the following open.
                if target_hit or i - entry_bar + 1 >= params.max_hold:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signals[i] = 1
            elif entries[i]:
                signals[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapReclaimParams()
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 100.0},
        index=pd.date_range("2020-01-01", periods=34),
    )
    data.iloc[21] = [98.0, 101.0, 97.0, 100.5, 250.0]
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    assert indicators["entry"].sum() == 1
    result = strategy.generate_signals(data, indicators, ctx, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[22:30] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()
    data.loc[data.index[23], ["high", "close"]] = [104.0, 103.0]
    result = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    expected[24:30] = 0
    assert np.array_equal(result["signal"].to_numpy(), expected)
    data.loc[data.index[21], "volume"] = 100.0
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
