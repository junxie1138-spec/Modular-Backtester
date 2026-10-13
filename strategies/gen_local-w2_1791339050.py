from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 12
    area_threshold: float = 0.15
    min_net_gap: float = 0.006
    profit_target: float = 0.04
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791339050"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.lookback) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.lookback < 2 or not 0 < params.area_threshold < 1:
            raise ValueError("Invalid signature parameters")
        if params.min_net_gap <= 0 or params.profit_target <= 0 or params.max_hold < 1:
            raise ValueError("Invalid entry or exit parameters")
        n = int(params.lookback)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        gap = (opening / close.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
        session = (close / opening - 1).replace([np.inf, -np.inf], np.nan)
        gap = gap.where(session.notna())
        session = session.where(gap.notna())
        # Second-order antisymmetric signature of cumulative (gap, session) returns.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, n):
            cross = gap.shift(lag) * session - session.shift(lag) * gap
            area = area + 0.5 * cross.rolling(n - lag, min_periods=n - lag).sum()
        scale = gap.abs().rolling(n).sum() * session.abs().rolling(n).sum()
        normalized = (2 * area / scale.where(scale > 0)).clip(-1, 1)
        net_gap = gap.rolling(n).sum()
        valid = normalized.notna() & net_gap.notna()
        out = pd.DataFrame(index=data.index)
        out["gap_return"] = gap.fillna(0.0)
        out["session_return"] = session.fillna(0.0)
        out["signature_area"] = area.fillna(0.0)
        out["normalized_area"] = normalized.fillna(0.0)
        out["net_gap"] = net_gap.fillna(0.0)
        out["valid"] = valid
        out["strength"] = (-normalized / (3 * params.area_threshold)).clip(0.25, 1).fillna(0.25)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        trigger = (indicators["valid"]
                   & (indicators["normalized_area"] <= -params.area_threshold)
                   & (indicators["net_gap"].abs() >= params.min_net_gap))
        candidates = (-np.sign(indicators["net_gap"])).where(trigger, 0).astype(int).to_numpy()
        strength = indicators["strength"].to_numpy()
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        side, entry_bar, entry_price, held_size = 0, -1, np.nan, 1.0
        for i in range(len(data)):
            if side:
                # Read the entry open only when that execution bar is reached.
                if i == entry_bar:
                    entry_price = opening[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    side = 0
                else:
                    # Gross close-marked gain; exits execute at the following open.
                    gain = side * (closing[i] / entry_price - 1)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                        side = 0
            elif candidates[i] and np.isfinite(closing[i]) and closing[i] > 0:
                side = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
                held_size = float(np.clip(strength[i], 0.25, 1.0))
            signals[i] = side
            sizes[i] = held_size if side else 1.0
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = Params(max_hold=3)
    index = pd.date_range("2020-01-01", periods=8)
    prices = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    features = pd.DataFrame({"valid": True, "normalized_area": -0.5,
                             "net_gap": -0.01, "strength": 0.7}, index=index)
    features.loc[index[4]:, "net_gap"] = 0.01
    result = GeneratedStrategy.generate_signals(None, prices, features, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1]
    prices.loc[index[1], "close"] = 105.0
    result = GeneratedStrategy.generate_signals(None, prices, features, None, p).data
    assert result["signal"].iloc[1:3].tolist() == [1, 0]
    assert result["size"].iloc[1] == 0.7
    features["net_gap"] = 0.01
    prices.loc[index[1], "close"] = 95.0
    result = GeneratedStrategy.generate_signals(None, prices, features, None, p).data
    assert result["signal"].iloc[1:3].tolist() == [-1, 0]
