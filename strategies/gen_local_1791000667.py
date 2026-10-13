from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreadthChangeParams:
    rank_window: int = 60
    profit_target: float = 0.008


class GeneratedStrategy(BaseStrategy[BreadthChangeParams]):
    strategy_id = "gen_local_1791000667"

    @classmethod
    def params_type(cls) -> type[BreadthChangeParams]:
        return BreadthChangeParams

    def warmup_bars(self, params: BreadthChangeParams) -> int:
        return max(2, int(params.rank_window)) + 12

    def indicators(self, data: pd.DataFrame, params: BreadthChangeParams) -> pd.DataFrame:
        window = max(2, int(params.rank_window))
        returns = data["close"].pct_change(fill_method=None)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        direction = np.sign(returns)
        recent = direction.rolling(3, min_periods=3).mean()
        established = direction.shift(3).rolling(9, min_periods=9).mean()
        change = recent - established
        percentile = change.rolling(window, min_periods=window).rank(method="average", pct=True)
        return pd.DataFrame({
            "return": returns,
            "direction": direction,
            "recent_breadth": recent,
            "established_breadth": established,
            "breadth_change": change,
            "change_percentile": percentile,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: BreadthChangeParams,
    ) -> SignalFrame:
        recent = indicators["recent_breadth"]
        established = indicators["established_breadth"]
        percentile = indicators["change_percentile"]
        ready = indicators.notna().all(axis=1)
        long_regime = ready & (recent >= 1.0 / 3.0) & (established <= -1.0 / 3.0) & (percentile >= 0.95)
        short_regime = ready & (recent <= -1.0 / 3.0) & (established >= 1.0 / 3.0) & (percentile <= 0.05)
        long_entry = long_regime & ~long_regime.shift(1, fill_value=False)
        short_entry = short_regime & ~short_regime.shift(1, fill_value=False)
        entries = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=np.int64)
        target = max(0.0001, float(params.profit_target))
        position = 0
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if position:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                # Close-observed exits execute at the following open.
                if not valid_entry or (np.isfinite(gain) and gain >= target) or age >= 2:
                    position = 0
                    decisions[i] = 0
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = position

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
