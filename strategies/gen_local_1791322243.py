from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    anchor_bars: int = 20
    volatility_bars: int = 10
    atr_bars: int = 10
    recovery_limit: int = 10
    hurdle_multiple: float = 1.0
    depth_multiple: float = 2.0
    breakeven_pct: float = 0.02
    trail_atr: float = 2.0
    max_hold_bars: int = 10
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791322243'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.anchor_bars, params.volatility_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.anchor_bars, params.volatility_bars, params.atr_bars) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if min(params.recovery_limit, params.max_hold_bars) < 1:
            raise ValueError('Holding and recovery limits must be positive.')
        if not all(np.isfinite(x) and x > 0 for x in (
            params.hurdle_multiple, params.depth_multiple,
            params.breakeven_pct, params.trail_atr, params.size
        )):
            raise ValueError('Thresholds and size must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['price_change'] = close.diff()
        out['sigma'] = out['price_change'].rolling(params.volatility_bars).std(ddof=0)
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars).mean()
        out['window_low'] = close.rolling(params.anchor_bars).min()
        out['window_high'] = close.rolling(params.anchor_bars).max()
        out['window_span'] = out['window_high'] - out['window_low']
        out['two_bar_change'] = close.diff(2)
        bar = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        for side, extreme in [('long', 'window_low'), ('short', 'window_high')]:
            event = close.eq(out[extreme]) & out['sigma'].gt(0)
            out[side + '_anchor'] = close.where(event).ffill()
            out[side + '_anchor_bar'] = bar.where(event).ffill()
            out[side + '_anchor_sigma'] = out['sigma'].where(event).ffill()
            out[side + '_anchor_depth'] = out['window_span'].where(event).ffill()
            out[side + '_age'] = bar - out[side + '_anchor_bar']
            out[side + '_hurdle'] = (
                params.hurdle_multiple * out[side + '_anchor_sigma']
                * np.sqrt(out[side + '_age'].clip(lower=0))
            )
            direction = 1 if side == 'long' else -1
            out[side + '_recovery'] = direction * (close - out[side + '_anchor'])
            out[side + '_candidate'] = (
                out[side + '_age'].between(1, params.recovery_limit)
                & out[side + '_anchor_depth'].ge(
                    params.depth_multiple * out[side + '_anchor_sigma']
                )
                & out[side + '_recovery'].gt(out[side + '_hurdle'])
                & (direction * out['two_bar_change']).gt(0)
                & out['atr'].gt(0)
            ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams
    ) -> SignalFrame:
        long_candidate = indicators['long_candidate'].fillna(False).astype(bool)
        short_candidate = indicators['short_candidate'].fillna(False).astype(bool)
        same_long_anchor = indicators['long_anchor_bar'].eq(indicators['long_anchor_bar'].shift(1))
        same_short_anchor = indicators['short_anchor_bar'].eq(indicators['short_anchor_bar'].shift(1))
        long_entry = (long_candidate & long_candidate.shift(1, fill_value=False) & same_long_anchor).to_numpy()
        short_entry = (short_candidate & short_candidate.shift(1, fill_value=False) & same_short_anchor).to_numpy()
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        stop_profit = np.nan
        armed = False
        held = 0
        for i in range(len(data)):
            if position:
                # Yesterday's decision fills at today's open.
                if np.isnan(entry):
                    if not (np.isfinite(opens[i]) and opens[i] > 0 and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                        position = 0
                        continue
                    entry = opens[i]
                    stop_profit = -params.trail_atr * atr[i - 1]
                    held = 0
                    armed = False
                held += 1
                if not (np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                    position = 0
                    entry = np.nan
                    continue
                profit = position * (closes[i] - entry)
                if profit >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed profit stops only rise, including for short positions.
                    stop_profit = max(stop_profit, 0.0, profit - params.trail_atr * atr[i])
                if profit <= stop_profit or held >= params.max_hold_bars:
                    position = 0
                    entry = np.nan
                    continue
            elif np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                if long_entry[i] and not short_entry[i]:
                    position = 1
                    entry = np.nan
                elif short_entry[i] and not long_entry[i]:
                    position = -1
                    entry = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': float(params.size)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=8)
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    for direction in (1, -1):
        prices = 100.0 + direction * np.array([0, 0, 0, 1, 3, 2, 0, 0], dtype=float)
        data = pd.DataFrame({
            'open': np.full(8, 100.0), 'high': prices + 1,
            'low': prices - 1, 'close': prices, 'volume': np.full(8, 1000.0)
        }, index=index)
        candidate = np.array([True, True, False, False, False, False, False, False])
        indicators = pd.DataFrame({
            'atr': np.ones(8),
            'long_candidate': candidate if direction == 1 else np.zeros(8, dtype=bool),
            'short_candidate': candidate if direction == -1 else np.zeros(8, dtype=bool),
            'long_anchor_bar': np.zeros(8), 'short_anchor_bar': np.zeros(8)
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, 0, direction, direction, direction, direction, direction, 0]
        assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
