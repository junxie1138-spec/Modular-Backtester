from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.45
    fractional_window: int = 40
    normalization_window: int = 20
    confirmation_bars: int = 3
    displacement_threshold: float = 0.8
    volatility_window: int = 20
    slow_volatility_window: int = 60
    contraction_ratio: float = 0.9
    target_volatility: float = 0.12
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 10

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must lie between zero and one')
        if min(self.fractional_window, self.normalization_window, self.volatility_window, self.slow_volatility_window, self.atr_window) < 2:
            raise ValueError('indicator windows must be at least two')
        if self.confirmation_bars < 1 or not 5 <= self.holding_bars <= 10:
            raise ValueError('invalid confirmation or holding period')
        if min(self.displacement_threshold, self.contraction_ratio, self.target_volatility, self.stop_atr) <= 0:
            raise ValueError('thresholds must be positive')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791208552'

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(max(200, params.fractional_window + params.normalization_window + params.confirmation_bars - 2, params.volatility_window + 1, params.slow_volatility_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        weights = np.ones(params.fractional_window, dtype=float)
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        fractional = pd.Series(0.0, index=data.index)
        for lag, weight in enumerate(weights):
            fractional = fractional + weight * close.shift(lag)
        fractional = fractional.replace([np.inf, -np.inf], np.nan)
        mean = fractional.rolling(params.normalization_window).mean()
        deviation = fractional.rolling(params.normalization_window).std(ddof=0)
        z = (fractional - mean) / deviation.replace(0.0, np.nan)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        fast_vol = returns.rolling(params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        slow_vol = returns.rolling(params.slow_volatility_window).std(ddof=0) * np.sqrt(252.0)
        previous_close = close.shift(1)
        true_range = pd.concat([(data['high'] - data['low']), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        regime_ma = close.rolling(200).mean()
        up = z.gt(params.displacement_threshold).rolling(params.confirmation_bars).sum().eq(params.confirmation_bars)
        down = z.lt(-params.displacement_threshold).rolling(params.confirmation_bars).sum().eq(params.confirmation_bars)
        contracted = fast_vol.lt(params.contraction_ratio * slow_vol)
        ready = z.notna() & regime_ma.notna() & atr.gt(0.0) & fast_vol.gt(0.0) & slow_vol.gt(0.0)
        direction = pd.Series(0, index=data.index, dtype=int)
        direction.loc[ready & contracted & up & close.gt(regime_ma)] = 1
        direction.loc[ready & contracted & down & close.lt(regime_ma)] = -1
        return pd.DataFrame({'fractional_close': fractional, 'fractional_mean': mean, 'fractional_std': deviation, 'fractional_z': z, 'realized_volatility': fast_vol, 'slow_volatility': slow_vol, 'atr': atr, 'regime_ma': regime_ma, 'confirmed_up': up, 'confirmed_down': down, 'contracted': contracted, 'entry_direction': direction}, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        directions = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        atrs = indicators['atr'].to_numpy(dtype=float)
        position = 0
        pending = False
        entry_atr = 0.0
        stop_price = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    stop_price = opens[i] - position * params.stop_atr * entry_atr
                    age = 0
                else:
                    position = 0
            if position:
                age += 1
                stopped = not np.isfinite(closes[i]) or position * (closes[i] - stop_price) <= 0.0
                if stopped or age >= params.holding_bars:
                    position = 0
                    signals[i] = 0
                    continue
            if position == 0 and directions[i] != 0 and np.isfinite(atrs[i]) and atrs[i] > 0.0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(directions[i])
                entry_atr = float(atrs[i])
                pending = True
            signals[i] = position
        volatility = indicators['realized_volatility'].replace([np.inf, -np.inf], np.nan)
        size = (params.target_volatility / volatility.where(volatility > 0.0)).clip(lower=0.05, upper=0.95).fillna(0.05)
        df = pd.DataFrame({'signal': signals, 'size': size.shift(1).fillna(0.05).astype(float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=14)
    for direction in (1, -1):
        data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 100.0, 'volume': 1000.0}, index=index)
        ind = pd.DataFrame({'entry_direction': [direction] + [0] * 13, 'atr': 1.0, 'realized_volatility': 0.2}, index=index)
        held = GeneratedStrategy.generate_signals(data, ind, None, params).data
        assert held['signal'].tolist() == [0] + [direction] * 10 + [0] * 3
        data.loc[index[2], 'close'] = 100.0 - direction * 3.0
        stopped = GeneratedStrategy.generate_signals(data, ind, None, params).data
        assert stopped['signal'].tolist() == [0, direction, direction] + [0] * 11
        assert stopped['size'].gt(0.0).all()
