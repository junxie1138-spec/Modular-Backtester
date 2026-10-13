from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    gap_shock: float = 1.0
    profit_target: float = 0.015


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1791303604"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        if not np.isfinite(params.gap_shock) or params.gap_shock <= 0:
            raise ValueError("gap_shock must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        prior_close = close.shift(1)
        out["return"] = (close / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["daily_vol"] = out["return"].rolling(20, min_periods=20).std(ddof=0)
        out["prior_vol"] = out["daily_vol"].shift(1)
        out["gap"] = (data["open"].where(data["open"] > 0) / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap_previous"] = out["gap"].shift(1)
        out["gap_first"] = out["gap"].shift(2)
        out["gap_pressure"] = -(out["gap"] + out["gap_previous"] + out["gap_first"])
        scale = out["prior_vol"].where(out["prior_vol"] > 0) * np.sqrt(3.0)
        out["shock_strength"] = out["gap_pressure"] / scale
        out["entry"] = (
            (out["gap"] < 0)
            & (out["gap_previous"] < 0)
            & (out["gap_first"] < 0)
            & (out["gap_previous"] < out["gap_first"])
            & (-out["gap"] <= 0.5 * -out["gap_previous"])
            & (out["shock_strength"] >= params.gap_shock)
            & close.notna()
            & out["daily_vol"].gt(0)
        ).fillna(False)
        out["annual_vol"] = out["daily_vol"] * np.sqrt(252.0)
        out["target_size"] = (
            (0.10 / out["annual_vol"].where(out["annual_vol"] > 0))
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=0.01, upper=0.95)
            .fillna(0.01)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.01, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        weights = indicators["target_size"].to_numpy(dtype=float)
        holding = False
        entry_bar = -1
        entry_price = np.nan
        held_size = 0.01
        for i in range(count):
            if holding:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = (
                    valid_entry
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if not valid_entry or target_hit or age >= 4:
                    holding = False
                else:
                    signals[i] = 1
                    sizes[i] = held_size
            elif entries[i]:
                weight = weights[i]
                if not np.isfinite(weight) or weight <= 0:
                    continue
                holding = True
                entry_bar = i + 1
                entry_price = np.nan
                held_size = float(np.clip(weight, 0.01, 0.95))
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": False, "target_size": 0.5}, index=index)
    indicators.loc[index[21], "entry"] = True
    strategy = GeneratedStrategy()
    params = ShockwaveParams()
    timed = strategy.generate_signals(data, indicators, None, params).data
    expected = np.zeros(30, dtype=int)
    expected[22:26] = 1
    assert np.array_equal(timed["signal"].to_numpy(), expected)
    assert (timed.loc[index[22:26], "size"] == 0.5).all()
    data.loc[index[23], "close"] = 102.0
    profit = strategy.generate_signals(data, indicators, None, params).data
    expected[24:26] = 0
    assert np.array_equal(profit["signal"].to_numpy(), expected)
    warmup = strategy.indicators(data.iloc[:10], params)
    assert not warmup["entry"].any()
    assert warmup["target_size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
