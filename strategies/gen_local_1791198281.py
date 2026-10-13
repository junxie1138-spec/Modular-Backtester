from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    lookback: int = 20
    shock_z: float = 1.8
    echo_max: float = 0.5
    atr_period: int = 14
    trail_atr: float = 1.0
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = 'gen_local_1791198281'

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(params.lookback + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_period < 1:
            raise ValueError('Lookbacks must be positive; lookback must be at least two.')
        if not np.isfinite(params.shock_z) or params.shock_z <= 0:
            raise ValueError('shock_z must be positive and finite.')
        if not np.isfinite(params.echo_max) or not 0 < params.echo_max < 1:
            raise ValueError('echo_max must be between zero and one.')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('trail_atr must be positive and finite.')
        if params.hold_days not in (1, 2):
            raise ValueError('hold_days must be one or two.')
        close = data['close'].astype(float)
        ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous = ret.shift(1)
        scale = ret.shift(2).rolling(params.lookback, min_periods=params.lookback).std(ddof=0).replace(0.0, np.nan)
        shock = previous.abs().div(scale)
        echo = ret.abs().div(previous.abs().replace(0.0, np.nan))
        prior_close = close.shift(1)
        tr = pd.concat([(data['high'] - data['low']).abs(), (data['high'] - prior_close).abs(), (data['low'] - prior_close).abs()], axis=1).max(axis=1, skipna=False)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        eligible = (shock >= params.shock_z) & (echo <= params.echo_max) & (ret * previous > 0) & (atr > 0)
        candidate = pd.Series(np.where(eligible, -np.sign(previous.fillna(0.0)), 0), index=data.index).astype(int)
        strength = ((shock / params.shock_z) * (1.0 - echo)).clip(lower=0.2, upper=1.0).replace([np.inf, -np.inf], np.nan).fillna(0.2)
        return pd.DataFrame({'return': ret, 'previous_return': previous, 'reference_volatility': scale, 'shock_z': shock, 'echo_ratio': echo, 'true_range': tr, 'atr': atr, 'candidate': candidate, 'strength': strength}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        returns = indicators['return'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].to_numpy(dtype=int)
        strengths = indicators['strength'].to_numpy(dtype=float)
        side = 0
        age = 0
        exposure = 1.0
        entry_atr = 0.0
        highest_close = -np.inf
        lowest_close = np.inf
        stop = 0.0
        for i in range(count):
            price = closes[i]
            if side:
                age += 1
                if not np.isfinite(price) or price <= 0:
                    side = 0
                    continue
                # The first close tracked is after the entry decision, on its fill bar.
                highest_close = max(highest_close, price)
                lowest_close = min(lowest_close, price)
                if side == 1:
                    stop = max(stop, highest_close - params.trail_atr * entry_atr)
                    stopped = price <= stop
                else:
                    # Mirrored ratchet for shorts: the lowest close plus an ATR buffer.
                    stop = min(stop, lowest_close + params.trail_atr * entry_atr)
                    stopped = price >= stop
                recovered = np.isfinite(returns[i]) and side * returns[i] > 0
                if stopped or recovered or age >= params.hold_days:
                    side = 0
                    continue
            elif candidates[i] and np.isfinite(price) and price > 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
                side = int(candidates[i])
                age = 0
                entry_atr = float(atrs[i])
                exposure = float(strengths[i])
                highest_close = -np.inf
                lowest_close = np.inf
                stop = -np.inf if side == 1 else np.inf
            if side:
                signals[i] = side
                sizes[i] = exposure
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = EchoParams()
    for direction in (-1, 1):
        returns = np.r_[np.tile([0.002, -0.002], 16), direction * 0.05, direction * 0.01, -direction * 0.01, 0.0]
        close = 100.0 * np.cumprod(np.r_[1.0, 1.0 + returns])
        data = pd.DataFrame({'open': np.r_[close[0], close[:-1]], 'high': close * 1.01, 'low': close * 0.99, 'close': close, 'volume': 1000000.0}, index=pd.date_range('2020-01-01', periods=len(close)))
        ind = strategy.indicators(data, params)
        result = strategy.generate_signals(data, ind, None, params).data
        assert ind['candidate'].iloc[-3] == -direction
        assert result['signal'].iloc[-3] == 0
        assert result['signal'].iloc[-2] == -direction
        assert result['signal'].iloc[-1] == 0
        assert result['signal'].isin([-1, 0, 1]).all()
        assert np.isfinite(result['size']).all() and (result['size'] > 0).all()
        prefix = data.iloc[:-1]
        prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
        pd.testing.assert_frame_equal(result.iloc[:-1], prefix_result)


if __name__ == '__main__':
    _self_check()
