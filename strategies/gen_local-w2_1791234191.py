from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_span: int = 8
    atr_window: int = 14
    vol_window: int = 20
    min_innovation: float = 0.10
    stop_atr: float = 1.5
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791234191'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.trend_span + 4, params.atr_window + 1, params.vol_window + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.trend_span, params.atr_window, params.vol_window) < 2:
            raise ValueError('Lookbacks must be at least two.')
        if not all(np.isfinite(x) and x > 0 for x in (params.min_innovation, params.stop_atr, params.target_vol)):
            raise ValueError('Threshold, stop multiple, and target volatility must be finite and positive.')
        close = data['close'].astype(float).where(data['close'] > 0)
        log_close = np.log(close)
        returns = log_close.diff()
        slope = returns.ewm(span=params.trend_span, adjust=False, min_periods=params.trend_span).mean()
        innovation = returns - slope.shift(1)
        previous_close = close.shift(1)
        tr = pd.concat([(data['high'] - data['low']).abs(), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        annual_vol = returns.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        scale = (atr / close).where(atr > 0)
        strength = innovation / scale
        prior = innovation.shift(1)
        earlier = innovation.shift(2)
        eligible = ((slope > 0) & (strength >= params.min_innovation) & (prior < 0) & (earlier > 0) & (prior.abs() < earlier.abs()) & (innovation.abs() < prior.abs()) & (annual_vol > 0) & (atr > 0))
        return pd.DataFrame({'log_close': log_close, 'returns': returns, 'slope': slope, 'innovation': innovation, 'previous_close': previous_close, 'true_range': tr, 'atr': atr, 'annual_vol': annual_vol, 'scale': scale, 'strength': strength, 'prior_innovation': prior, 'earlier_innovation': earlier, 'eligible': eligible.fillna(False)}, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        vol = indicators['annual_vol'].to_numpy(dtype=float)
        strength = indicators['strength'].to_numpy(dtype=float)
        eligible = indicators['eligible'].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        age = 0
        stop = 0.0
        allocation = 1.0
        for i in range(n):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    age = 0
                    stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= 2:
                    active = False
                else:
                    signal[i] = 1
                    sizes[i] = allocation
                continue
            if eligible[i] and np.isfinite(vol[i]) and vol[i] > 0 and np.isfinite(strength[i]) and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                conviction = float(np.clip(strength[i] / (3.0 * params.min_innovation), 0.0, 1.0))
                allocation = float(np.clip(params.target_vol / vol[i] * conviction, 0.05, 1.0))
                signal[i] = 1
                sizes[i] = allocation
                pending = True
        df = pd.DataFrame({'signal': signal, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'open': [100.0, 110.0, 110.0, 110.0, 110.0], 'high': 112.0, 'low': 99.0, 'close': [100.0, 107.0, 110.0, 110.0, 110.0], 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'atr': 1.0, 'annual_vol': 0.24, 'strength': 0.15, 'eligible': [True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    stopped = strategy.generate_signals(data, ind, None, params).data
    assert stopped['signal'].tolist() == [0, 1, 0, 0, 0]
    assert np.isclose(stopped['size'].iloc[1], 0.25)
    data['close'] = 110.0
    expired = strategy.generate_signals(data, ind, None, params).data
    assert expired['signal'].tolist() == [0, 1, 1, 0, 0]
    assert (expired['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
