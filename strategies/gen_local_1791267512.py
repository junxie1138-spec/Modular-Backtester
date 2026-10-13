from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    min_depth: float = 0.006
    max_depth: float = 0.035
    max_damage_ratio: float = 0.65
    min_prior_advance: float = 0.015
    profit_target: float = 0.035
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791267512"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["peak"] = close.shift(1).rolling(8, min_periods=8).max()
        out["previous_peak"] = out["peak"].shift(1)
        out["depth"] = (out["peak"] - close) / out["peak"]
        out["previous_depth"] = out["depth"].shift(1)
        out["downside_probe"] = ((opening - data["low"]) / opening).clip(lower=0)
        out["selling_effort"] = out["downside_probe"].rolling(8, min_periods=8).sum()
        out["previous_effort"] = out["selling_effort"].shift(1)
        out["damage_ratio"] = out["depth"] / out["selling_effort"].replace(0, np.nan)
        out["baseline"] = close.shift(8)
        out["prior_advance"] = out["peak"] / out["baseline"] - 1.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        qualified = (
            indicators["depth"].between(params.min_depth, params.max_depth)
            & (indicators["depth"] < indicators["previous_depth"])
            & (indicators["peak"] == indicators["previous_peak"])
            & (indicators["selling_effort"] > indicators["previous_effort"])
            & (indicators["damage_ratio"] <= params.max_damage_ratio)
            & (indicators["prior_advance"] >= params.min_prior_advance)
            & (data["close"] > indicators["baseline"])
            & (data["close"] > data["open"])
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        entry_bar = -1
        hold_limit = max(1, int(params.max_hold_bars))

        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                timed_out = i - entry_bar + 1 >= hold_limit
                if target_hit or timed_out:
                    active = False
                    entry_price = np.nan
                else:
                    signals[i] = 1
            elif qualified[i]:
                active = True
                entry_bar = i + 1
                signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame(
        {
            "depth": 0.01,
            "previous_depth": 0.02,
            "peak": 102.0,
            "previous_peak": 102.0,
            "selling_effort": 0.05,
            "previous_effort": 0.04,
            "damage_ratio": 0.2,
            "baseline": 98.0,
            "prior_advance": 0.04,
        },
        index=index,
    )
    data.loc[index[0], "close"] = 100.5
    strategy = GeneratedStrategy()
    params = StrategyParams()
    timed = strategy.generate_signals(data, ind, None, params).data["signal"]
    assert timed.iloc[0] == 0
    assert timed.iloc[1:19].eq(1).all()
    assert timed.iloc[19:].eq(0).all()
    data.loc[index[3], "close"] = 104.0
    ind.loc[index[1]:, "depth"] = np.nan
    targeted = strategy.generate_signals(data, ind, None, params).data["signal"]
    assert targeted.iloc[1:4].eq(1).all()
    assert targeted.iloc[4:].eq(0).all()
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
