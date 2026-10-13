from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalAnalogParams:
    history: int = 126
    entropy_window: int = 16
    entropy_limit: float = 0.75
    vol_window: int = 20
    min_matches: int = 10
    edge_z: float = 0.18
    target_pct: float = 0.008
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[OrdinalAnalogParams]):
    strategy_id = "gen_local_1791223702"

    @classmethod
    def params_type(cls):
        return OrdinalAnalogParams

    def warmup_bars(self, params: OrdinalAnalogParams) -> int:
        return max(params.history + 4, params.entropy_window + 4, params.vol_window + 1)

    def indicators(self, data: pd.DataFrame, params: OrdinalAnalogParams) -> pd.DataFrame:
        returns = data["close"].pct_change(fill_method=None)
        a, b, c = returns, returns.shift(1), returns.shift(2)
        pattern = ((a > b).astype(int) + 2 * (a > c).astype(int) + 4 * (b > c).astype(int)).astype(float)
        pattern = pattern.where(a.notna() & b.notna() & c.notna())

        entropy = pd.Series(0.0, index=data.index)
        edge = pd.Series(0.0, index=data.index)
        matches = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            motif = pattern.eq(code)
            frequency = motif.astype(float).rolling(params.entropy_window, min_periods=params.entropy_window).mean()
            entropy = entropy - frequency * np.log(frequency.where(frequency > 0, 1.0))

            prior_motif = motif.shift(1, fill_value=False).astype(float)
            count = prior_motif.rolling(params.history, min_periods=params.history).sum()
            payoff = (returns * prior_motif).rolling(params.history, min_periods=params.history).sum()
            edge = edge.where(~motif, payoff.div(count.where(count > 0)).fillna(0.0))
            matches = matches.where(~motif, count.fillna(0.0))

        complete = pattern.notna().rolling(params.entropy_window, min_periods=params.entropy_window).sum().eq(params.entropy_window)
        entropy = (entropy / np.log(6.0)).where(complete, 1.0).fillna(1.0)
        volatility = returns.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0)
        strength = (edge.abs() / volatility.where(volatility > 0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({"entropy": entropy, "edge": edge, "matches": matches, "strength": strength}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalAnalogParams) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        edge = indicators["edge"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        eligible = ((indicators["entropy"] <= params.entropy_limit) & (indicators["matches"] >= params.min_matches) & (indicators["strength"] >= params.edge_z)).to_numpy(dtype=bool)
        eligible[:self.warmup_bars(params)] = False
        scaled_size = np.clip(strength * 1.5, 0.25, 0.95)
        entry_bar = -1
        entry_price = 0.0
        entry_size = 0.25

        for i in range(n):
            active = raw[i - 1] if i else 0
            if active:
                if i == 1 or raw[i - 2] != active:
                    entry_bar = i
                    entry_price = opens[i]
                    entry_size = sizes[i - 1]
                gained = active * (closes[i] / entry_price - 1.0)
                if gained >= params.target_pct or i - entry_bar + 1 >= params.hold_bars:
                    raw[i] = 0
                else:
                    raw[i] = active
                    sizes[i] = entry_size
            elif eligible[i]:
                raw[i] = 1 if edge[i] > 0 else -1
                sizes[i] = float(scaled_size[i])

        df = pd.DataFrame({"signal": raw, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
