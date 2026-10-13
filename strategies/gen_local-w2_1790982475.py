from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    streak_days: int = 2
    recovery_fraction: float = 0.75
    atr_period: int = 14
    trail_atr: float = 0.9
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1790982475'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(210, params.atr_period + 1, params.streak_days + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.streak_days < 1 or params.atr_period < 1:
            raise ValueError('Lookbacks must be positive.')
        if not np.isfinite(params.recovery_fraction) or params.recovery_fraction <= 0:
            raise ValueError('Recovery fraction must be finite and positive.')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('ATR multiplier must be finite and positive.')
        if params.hold_days not in (1, 2):
            raise ValueError('Holding period must be one or two sessions.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous = close.shift(1)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['ma_slope'] = out['ma200'].diff(10)
        gap_loss = previous - data['open']
        out['recovery'] = ((close - data['open']) / gap_loss.where(gap_loss > 0)).replace([np.inf, -np.inf], np.nan)
        repair = (gap_loss > 0) & (out['recovery'] >= params.recovery_fraction)
        out['repair_streak'] = repair.astype(int).groupby((~repair).cumsum()).cumsum()
        declining = close.diff().lt(0)
        out['decline_streak'] = declining.astype(int).groupby((~declining).cumsum()).cumsum()
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous.notna())
        out['atr'] = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        out['ready'] = (
            np.isfinite(close) & (close > 0)
            & np.isfinite(out['ma200']) & np.isfinite(out['ma_slope'])
            & np.isfinite(out['atr']) & (out['atr'] > 0)
        )
        out['regime_ok'] = out['ready'] & close.gt(out['ma200'])
        continuation = out['ma_slope'].gt(0) & out['repair_streak'].eq(params.streak_days)
        rebound = (
            out['ma_slope'].le(0)
            & out['decline_streak'].shift(1).ge(params.streak_days)
            & repair & close.gt(previous)
        )
        out['entry'] = (out['regime_ok'] & (continuation | rebound)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        allowed = indicators['regime_ok'].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0
            if active:
                age += 1
                if valid:
                    # First in-trade close is the bar after the entry decision.
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atrs[i])
                if not valid or not allowed[i] or closes[i] <= stop or age >= params.hold_days:
                    active = False
                else:
                    signals[i] = 1
                # An exit decision cannot also reopen the position.
                continue
            if valid and allowed[i] and entries[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
        df['signal'] = signals
        df['size'] = 1.0
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 98.0, 'close': [100.0, 101.0, 99.0, 100.0, 101.0, 102.0, 100.0], 'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'atr': 1.0, 'entry': [True, False, False, True, False, False, False], 'regime_ok': True}, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=210)
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1, 0]
    derived.loc[index[1], 'regime_ok'] = False
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 1, 1, 0]
    assert result.index.equals(data.index) and result['size'].gt(0).all()
    flat = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=pd.date_range('2020-01-01', periods=240))
    calculated = strategy.indicators(flat, params)
    assert not calculated['entry'].any()
    assert not calculated['ready'].iloc[:209].any()
    assert strategy.warmup_bars(params) >= 210


if __name__ == '__main__':
    _self_check()
