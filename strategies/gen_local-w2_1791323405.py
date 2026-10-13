from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRecoveryParams:
    drawdown_window: int = 60
    gap_block: int = 5
    gap_scale_window: int = 20
    min_drawdown: float = 0.025
    max_drawdown: float = 0.15
    contraction_limit: float = 0.70
    min_gap_z: float = 0.35
    profit_target: float = 0.045
    max_hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapRecoveryParams]):
    strategy_id = 'gen_local-w2_1791323405'

    @classmethod
    def params_type(cls) -> type[GapRecoveryParams]:
        return GapRecoveryParams

    def warmup_bars(self, params: GapRecoveryParams) -> int:
        return int(max(params.drawdown_window, 2 * params.gap_block + 2, params.gap_scale_window + 2))

    def indicators(self, data: pd.DataFrame, params: GapRecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1).replace(0.0, np.nan)
        out['gap'] = (data['open'] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out['downside_gap'] = (-out['gap']).clip(lower=0.0)
        out['recent_damage'] = out['downside_gap'].shift(1).rolling(params.gap_block, min_periods=params.gap_block).mean()
        out['earlier_damage'] = out['downside_gap'].shift(params.gap_block + 1).rolling(params.gap_block, min_periods=params.gap_block).mean()
        out['contraction'] = out['recent_damage'] / out['earlier_damage'].replace(0.0, np.nan)
        out['gap_scale'] = out['gap'].shift(1).rolling(params.gap_scale_window, min_periods=params.gap_scale_window).std(ddof=0)
        out['gap_z'] = out['gap'] / out['gap_scale'].replace(0.0, np.nan)
        out['peak'] = data['close'].rolling(params.drawdown_window, min_periods=params.drawdown_window).max()
        out['drawdown'] = 1.0 - data['close'] / out['peak'].replace(0.0, np.nan)
        out['acceptance'] = (data['close'] >= data['open']) & data['close'].notna() & data['open'].notna()
        out['strength'] = (0.6 * (out['gap_z'] / 2.0).clip(0.0, 1.0) + 0.4 * (1.0 - out['contraction']).clip(0.0, 1.0)).fillna(0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRecoveryParams) -> SignalFrame:
        entry = (
            indicators['drawdown'].between(params.min_drawdown, params.max_drawdown)
            & indicators['contraction'].le(params.contraction_limit)
            & indicators['earlier_damage'].gt(0.0)
            & indicators['gap_z'].ge(params.min_gap_z)
            & indicators['gap'].gt(0.0)
            & indicators['acceptance'].fillna(False)
        ).fillna(False).to_numpy(dtype=bool)
        strength = indicators['strength'].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        active = False
        pending_fill = False
        entry_price = np.nan
        held_bars = 0
        position_size = params.min_size

        for i in range(len(data)):
            if active:
                if pending_fill:
                    entry_price = opens[i]
                    pending_fill = False
                    held_bars = 0
                held_bars += 1
                sizes[i] = position_size
                target_hit = bool(np.isfinite(entry_price) and entry_price > 0.0 and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target))
                if target_hit or held_bars >= params.max_hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif entry[i]:
                active = True
                pending_fill = True
                position_size = float(params.min_size + (params.max_size - params.min_size) * strength[i])
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
