from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_bars: int = 10
    smoothing_bars: int = 3
    entry_threshold: float = 0.45


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790991893"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return max(2, int(params.graph_bars)) + max(1, int(params.smoothing_bars))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        lookback = max(2, int(params.graph_bars))
        smoothing = max(1, int(params.smoothing_bars))
        close = data["close"].astype(float)
        price = np.log(close.where(np.isfinite(close) & close.gt(0)))
        valid = price.notna().astype(float).rolling(
            lookback + 1, min_periods=lookback + 1
        ).sum().eq(lookback + 1)
        signed_edges = pd.Series(0.0, index=data.index)
        total_edges = pd.Series(0.0, index=data.index)
        distant_edges = pd.Series(0.0, index=data.index)

        # Natural visibility above and below price makes the topology symmetric.
        # Each lag is vectorized over all bars; intermediate prices must lie
        # strictly on one side of the line connecting the edge endpoints.
        for lag in range(2, lookback + 1):
            past = price.shift(lag)
            upper_visible = price.notna() & past.notna()
            lower_visible = upper_visible.copy()
            for offset in range(1, lag):
                intermediate = price.shift(offset)
                line = price + (past - price) * (float(offset) / float(lag))
                upper_visible &= intermediate.lt(line)
                lower_visible &= intermediate.gt(line)
            edge_count = upper_visible.astype(float) + lower_visible.astype(float)
            weight = float(lag) / float(lookback)
            direction = np.sign(price - past).fillna(0.0)
            signed_edges += weight * edge_count * direction
            total_edges += weight * edge_count
            if lag >= max(2, (lookback + 1) // 2):
                distant_edges += edge_count

        signed_edges = signed_edges.where(valid)
        total_edges = total_edges.where(valid)
        distant_edges = distant_edges.where(valid)
        smoothed_signed = signed_edges.rolling(
            smoothing, min_periods=smoothing
        ).mean()
        smoothed_total = total_edges.rolling(
            smoothing, min_periods=smoothing
        ).mean()
        smoothed_distant = distant_edges.rolling(
            smoothing, min_periods=smoothing
        ).mean()
        strength = smoothed_signed.div(smoothed_total.replace(0.0, np.nan))
        strength = strength.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ready = smoothed_total.notna() & smoothed_distant.notna()

        return pd.DataFrame({
            "log_close": price,
            "signed_edges": signed_edges,
            "total_edges": total_edges,
            "distant_edges": distant_edges,
            "smoothed_signed_edges": smoothed_signed,
            "smoothed_total_edges": smoothed_total,
            "smoothed_distant_edges": smoothed_distant,
            "topology_strength": strength,
            "ready": ready,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        threshold = float(np.clip(params.entry_threshold, 0.01, 1.0))
        eligible = indicators["ready"].fillna(False).astype(bool) & (
            indicators["smoothed_distant_edges"].fillna(0.0) >= 1.0
        )
        strength = indicators["topology_strength"].fillna(0.0)
        condition = pd.Series(0, index=data.index, dtype=int)
        condition.loc[eligible & strength.ge(threshold)] = 1
        condition.loc[eligible & strength.le(-threshold)] = -1
        confirmed = condition.ne(0) & condition.eq(condition.shift(1))
        conditions = condition.to_numpy(dtype=int)
        confirmations = confirmed.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0

        for bar in range(len(data)):
            direction = int(conditions[bar])
            if position != 0:
                # Exit only on the opposite qualifying entry condition.
                # Neutral and unavailable observations preserve exposure.
                if direction == -position:
                    position = 0
            elif confirmations[bar]:
                position = direction
            signals[bar] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
