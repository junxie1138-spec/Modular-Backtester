from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    corridor_bars: int = 12
    position_ceiling: float = 0.45
    minimum_low_lift: float = 0.10
    pressure_bars: int = 3
    atr_bars: int = 14
    stop_atr: float = 1.3
    holding_bars: int = 4
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791237785"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.corridor_bars, params.pressure_bars, params.atr_bars) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.corridor_bars, params.pressure_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between 3 and 5.")
        if not (0 < params.position_ceiling < 1 and params.minimum_low_lift >= 0
                and params.stop_atr > 0 and params.entry_slippage_bps >= 0):
            raise ValueError("Invalid entry or stop parameters.")
        out = pd.DataFrame(index=data.index)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["previous_close"] = data["close"].shift(1)
        out["previous_range"] = (out["previous_high"] - out["previous_low"]).where(
            out["previous_high"] > out["previous_low"]
        )
        out["corridor_high"] = data["high"].shift(1).rolling(params.corridor_bars).max()
        out["corridor_low"] = data["low"].shift(1).rolling(params.corridor_bars).min()
        out["corridor_width"] = (out["corridor_high"] - out["corridor_low"]).where(
            out["corridor_high"] > out["corridor_low"]
        )
        out["range_position"] = (data["close"] - out["corridor_low"]) / out["corridor_width"]
        out["low_lift"] = (data["low"] - out["previous_low"]) / out["previous_range"]
        out["high_change"] = (data["high"] - out["previous_high"]) / out["previous_range"]
        out["signed_volume"] = np.sign(data["close"] - data["open"]) * data["volume"].clip(lower=0)
        out["volume_sum"] = data["volume"].clip(lower=0).rolling(params.pressure_bars).sum()
        out["signed_volume_sum"] = out["signed_volume"].rolling(params.pressure_bars).sum()
        out["pressure"] = out["signed_volume_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out["previous_pressure"] = out["pressure"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["range_agreement"] = (
            (out["low_lift"] >= params.minimum_low_lift)
            & (out["high_change"] <= 0)
            & (out["range_position"] > 0)
            & (out["range_position"] <= params.position_ceiling)
        )
        out["turnover_agreement"] = (out["pressure"] > 0) & (out["previous_pressure"] <= 0)
        out["entry"] = (
            out["range_agreement"] & out["turnover_agreement"]
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        fixed_stop = np.nan
        frozen_atr = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Use this entry bar's open only at its close; ATR was frozen earlier.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    entry_price = opens[i] * (1.0 + params.entry_slippage_bps / 10000.0)
                    fixed_stop = entry_price - params.stop_atr * frozen_atr
                if not np.isfinite(closes[i]) or closes[i] < fixed_stop or i - entry_bar + 1 >= params.holding_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                entry_bar = i + 1
                frozen_atr = atr[i]
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 112.0, "low": 96.0,
        "close": [100.0, 100.0, 110.0, 98.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"entry": [True] + [False] * 7, "atr": 2.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], "close"] = 97.0
    ind.loc[index[2]:, "atr"] = 100.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
