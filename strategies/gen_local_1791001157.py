from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RejectionParams:
    atr_window: int = 14
    volatility_window: int = 20
    shock_threshold: float = 1.5
    displacement_ceiling: float = 0.7
    rejection_threshold: float = 0.2
    target_daily_vol: float = 0.008
    stop_atr: float = 1.2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RejectionParams]):
    strategy_id = 'gen_local_1791001157'

    @classmethod
    def params_type(cls) -> type[RejectionParams]:
        return RejectionParams

    def warmup_bars(self, params: RejectionParams) -> int:
        return max(2, params.atr_window, params.volatility_window) + 2

    def indicators(self, data: pd.DataFrame, params: RejectionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1).where(lambda x: x > 0)
        candle_range = (data['high'] - data['low']).clip(lower=0)
        true_range = pd.concat([
            candle_range,
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr_window = max(2, int(params.atr_window))
        volatility_window = max(2, int(params.volatility_window))
        out['true_range'] = true_range
        out['atr'] = true_range.rolling(atr_window, min_periods=atr_window).mean()
        out['prior_atr'] = out['atr'].shift(1)
        out['returns'] = data['close'].div(previous_close).sub(1)
        out['realized_vol'] = out['returns'].rolling(
            volatility_window, min_periods=volatility_window
        ).std(ddof=0)
        out['range_vol'] = out['prior_atr'].div(previous_close)
        out['shock'] = true_range.div(out['prior_atr'].where(out['prior_atr'] > 0))
        out['displacement_ratio'] = out['realized_vol'].div(
            out['range_vol'].where(out['range_vol'] > 0)
        )
        body_top = data[['open', 'close']].max(axis=1)
        body_bottom = data[['open', 'close']].min(axis=1)
        out['upper_wick'] = (data['high'] - body_top).clip(lower=0)
        out['lower_wick'] = (body_bottom - data['low']).clip(lower=0)
        out['rejection'] = (out['lower_wick'] - out['upper_wick']).div(
            candle_range.where(candle_range > 0)
        )
        out['risk_vol'] = out[['realized_vol', 'range_vol']].max(axis=1)
        out['conviction'] = (
            out['rejection'].abs().clip(0, 1)
            * (1 - out['displacement_ratio'].div(max(params.displacement_ceiling, 1e-8))).clip(0, 1)
            * (out['shock'].div(max(params.shock_threshold, 1e-8)) - 1).clip(0, 1)
        )
        out['entry_size'] = (
            max(params.target_daily_vol, 1e-8)
            / out['risk_vol'].where(out['risk_vol'] > 0)
            * out['conviction']
        ).clip(lower=1e-6, upper=max(params.max_size, 1e-6))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RejectionParams,
    ) -> SignalFrame:
        required = ['atr', 'shock', 'displacement_ratio', 'rejection', 'risk_vol', 'entry_size']
        valid = indicators[required].notna().all(axis=1)
        eligible = (
            valid
            & (indicators['atr'] > 0)
            & (indicators['risk_vol'] > 0)
            & (indicators['shock'] >= params.shock_threshold)
            & (indicators['displacement_ratio'] < params.displacement_ceiling)
            & (indicators['rejection'].abs() >= params.rejection_threshold)
        )
        candidates = np.where(eligible, np.sign(indicators['rejection'].fillna(0)), 0).astype(int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        proposed_sizes = indicators['entry_size'].fillna(1e-6).to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        decision_sizes = np.full(len(data), 1e-6, dtype=float)
        position = 0
        pending_entry = False
        entry_price = 0.0
        frozen_atr = 0.0
        held_bars = 0
        active_size = 1e-6
        stop_multiple = max(params.stop_atr, 1e-8)

        for i in range(len(data)):
            if position != 0:
                if pending_entry:
                    entry_price = opens[i]
                    frozen_atr = atr[i - 1]
                    pending_entry = False
                held_bars += 1
                invalid = (
                    not np.isfinite(entry_price)
                    or entry_price <= 0
                    or not np.isfinite(frozen_atr)
                    or frozen_atr <= 0
                    or not np.isfinite(closes[i])
                )
                stopped = (
                    closes[i] < entry_price - stop_multiple * frozen_atr
                    if position == 1
                    else closes[i] > entry_price + stop_multiple * frozen_atr
                )
                if invalid or stopped or held_bars >= 2:
                    position = 0
                    held_bars = 0
                    continue
                decisions[i] = position
                decision_sizes[i] = active_size
            elif candidates[i] != 0:
                position = int(candidates[i])
                pending_entry = True
                held_bars = 0
                active_size = float(proposed_sizes[i])
                decisions[i] = position
                decision_sizes[i] = active_size

        df = pd.DataFrame({'signal': decisions, 'size': decision_sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1e-6).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
