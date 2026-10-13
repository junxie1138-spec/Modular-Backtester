from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PropagationParams:
    entry_z: float = 0.75
    profit_target: float = 0.04


class GeneratedStrategy(BaseStrategy[PropagationParams]):
    strategy_id = "gen_local_1791322522"

    @classmethod
    def params_type(cls) -> type[PropagationParams]:
        return PropagationParams

    def warmup_bars(self, params: PropagationParams) -> int:
        return 60

    def indicators(self, data: pd.DataFrame, params: PropagationParams) -> pd.DataFrame:
        close = data["close"].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        ma = close.rolling(40, min_periods=40).mean()
        scale = close.rolling(40, min_periods=40).std(ddof=0)
        z = ((close - ma) / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)
        mass = z.clip(lower=0).rolling(5, min_periods=5).sum()
        prior_mass = mass.shift(5)
        reproduction = mass / prior_mass.where(prior_mass > 0)
        susceptible = z.lt(0).astype(float).where(z.notna()).rolling(20, min_periods=20).mean()
        slope = ma.diff(5)
        entry = (
            z.ge(params.entry_z)
            & z.le(2.0)
            & reproduction.between(1.10, 1.80)
            & susceptible.between(0.20, 0.60)
            & slope.gt(0)
        ).fillna(False)
        return pd.DataFrame({
            "ma": ma,
            "scale": scale,
            "z": z,
            "positive_mass": mass,
            "prior_mass": prior_mass,
            "reproduction": reproduction,
            "susceptible": susceptible,
            "ma_slope": slope,
            "entry": entry,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PropagationParams) -> SignalFrame:
        if not np.isfinite(params.entry_z) or not 0 < params.entry_z <= 2.0:
            raise ValueError("entry_z must be finite and in (0, 2]")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if active:
                # The preceding close's entry decision fills at this bar's open.
                if age == 0:
                    entry_price = opens[i]
                age += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if not valid_entry or target_hit or age >= 18:
                    active = False
                    age = 0
                else:
                    signal[i] = 1
            elif entry[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_price = np.nan
                age = 0
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = PropagationParams()
    index = pd.date_range("2020-01-01", periods=22, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [True] + [False] * 21}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] * 3
    data.loc[index[1]:, ["open", "high", "low", "close"]] = 120.0
    data.loc[index[3]:, ["high", "close"]] = 126.0
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1] + [0] * 18
    assert strategy.indicators(data, params)["entry"].eq(False).all()
