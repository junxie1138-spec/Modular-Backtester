from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReclaimParams:
    gap_depth_min: float = 0.25
    body_strength: float = 0.60
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapReclaimParams]):
    strategy_id = "gen_local-w2_1790993312"

    @classmethod
    def params_type(cls) -> type[GapReclaimParams]:
        return GapReclaimParams

    def warmup_bars(self, params: GapReclaimParams) -> int:
        return 6

    def indicators(self, data: pd.DataFrame, params: GapReclaimParams) -> pd.DataFrame:
        clean = data[['open', 'high', 'low', 'close']].replace([np.inf, -np.inf], np.nan)
        prior_open = clean['open'].shift(1)
        prior_close = clean['close'].shift(1)
        prior_body = prior_close - prior_open
        midpoint = (prior_close + prior_open) / 2.0
        mean_range = (clean['high'] - clean['low']).rolling(5, min_periods=5).mean().shift(1)
        gap = clean['open'] - prior_close
        gap_depth = -gap / prior_body.where(prior_body > 0.0)
        body_ratio = prior_body / mean_range.where(mean_range > 0.0)
        ready = clean.notna().all(axis=1) & prior_open.notna() & prior_close.notna() & mean_range.notna()
        entry = (
            ready
            & (clean['open'] > 0.0)
            & (clean['close'] > 0.0)
            & (prior_body > 0.0)
            & (body_ratio >= params.body_strength)
            & (gap < 0.0)
            & (gap_depth >= params.gap_depth_min)
            & (gap_depth < 1.0)
            & (clean['low'] < midpoint)
            & (clean['close'] > midpoint)
            & (clean['close'] > clean['open'])
            & (clean['close'] < prior_close)
        )
        out = pd.DataFrame({
            'prior_open': prior_open,
            'prior_close': prior_close,
            'prior_body': prior_body,
            'body_midpoint': midpoint,
            'mean_range': mean_range,
            'gap': gap,
            'gap_depth': gap_depth,
            'body_ratio': body_ratio,
            'ready': ready,
            'entry': entry.fillna(False),
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapReclaimParams,
    ) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=bool)
        desired = np.zeros(len(data), dtype=int)
        hold_limit = min(2, max(1, int(params.max_hold_bars)))
        target = max(0.0001, float(params.profit_target))
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if entry_bar >= 0:
                # The preceding close's decision fills at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    entry_bar = -1
                    entry_price = np.nan
                    continue
                held_bars = i - entry_bar + 1
                # A close-observed target schedules an exit at the next open.
                profit_exit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + target)
                time_exit = held_bars >= hold_limit
                if profit_exit or time_exit:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    desired[i] = 1
            elif entries[i]:
                desired[i] = 1
                entry_bar = i + 1

        df = pd.DataFrame({'signal': desired, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
