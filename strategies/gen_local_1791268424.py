from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 20
    min_breadth: float = 2.6
    min_coherence: float = 0.6
    target_vol: float = 0.12
    atr_period: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 8
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local_1791268424'

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(200, 16 + params.energy_window, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 2 or params.atr_period < 1 or params.hold_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive; energy_window must be at least two.')
        if not (1.0 <= params.min_breadth <= 4.0 and 0.0 < params.min_coherence <= 1.0):
            raise ValueError('Invalid breadth or coherence threshold.')
        if not (0.0 < params.target_vol and 0.0 < params.stop_atr and 0.0 < params.max_size <= 1.0):
            raise ValueError('Volatility, stop distance and position limit must be positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_return'] = np.log(close.where(close > 0)).diff()
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_period, min_periods=params.atr_period).mean()
        energy_columns = []
        coefficient_columns = []
        for half in (1, 2, 4, 8):
            block = out['log_return'].rolling(half, min_periods=half).mean()
            out[f'block_{half}'] = block
            coefficient = f'haar_{half}'
            energy = f'energy_{half}'
            out[coefficient] = np.sqrt(half / 2.0) * (block - block.shift(half))
            out[energy] = out[coefficient].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
            coefficient_columns.append(coefficient)
            energy_columns.append(energy)
        energy = out[energy_columns]
        out['total_energy'] = energy.sum(axis=1, min_count=4)
        shares = energy.div(out['total_energy'].where(out['total_energy'] > 0), axis=0)
        for half, column in zip((1, 2, 4, 8), energy_columns):
            out[f'share_{half}'] = shares[column]
        out['breadth'] = 1.0 / shares.pow(2).sum(axis=1, min_count=4).replace(0.0, np.nan)
        polarity = np.sign(out[coefficient_columns])
        polarity.columns = energy_columns
        out['coherence'] = (shares * polarity).sum(axis=1, min_count=4)
        out['realized_vol'] = out['log_return'].rolling(params.energy_window, min_periods=params.energy_window).std(ddof=0) * np.sqrt(252.0)
        out['wavelet_vol'] = np.sqrt(out['total_energy'] / 4.0 * 252.0)
        out['risk_vol'] = out[['realized_vol', 'wavelet_vol']].max(axis=1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        close = data['close'].to_numpy(dtype=float)
        ma = indicators['ma200'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        coherence = indicators['coherence'].to_numpy(dtype=float)
        breadth = indicators['breadth'].to_numpy(dtype=float)
        risk = indicators['risk_vol'].to_numpy(dtype=float)
        ready = np.isfinite(close) & np.isfinite(ma) & np.isfinite(atr) & np.isfinite(coherence) & np.isfinite(breadth) & np.isfinite(risk)
        ready &= (close > 0.0) & (atr > 0.0) & (risk > 0.0)
        entry = np.zeros(len(data), dtype=int)
        qualified = ready & (breadth >= params.min_breadth)
        entry[qualified & (close > ma) & (coherence >= params.min_coherence)] = 1
        entry[qualified & (close < ma) & (coherence <= -params.min_coherence)] = -1
        signal = np.zeros(len(data), dtype=int)
        position = 0
        held = 0
        extreme = np.nan
        stop = np.nan
        for i in range(len(data)):
            if position:
                held += 1
                if not ready[i]:
                    position = 0
                else:
                    # The first in-trade close is the bar after the entry decision.
                    if position == 1:
                        extreme = close[i] if not np.isfinite(extreme) else max(extreme, close[i])
                        candidate = extreme - params.stop_atr * atr[i]
                        stop = candidate if not np.isfinite(stop) else max(stop, candidate)
                        stopped = close[i] <= stop
                        regime_exit = close[i] <= ma[i]
                    else:
                        # Mirror the high-water trail for shorts using the lowest close.
                        extreme = close[i] if not np.isfinite(extreme) else min(extreme, close[i])
                        candidate = extreme + params.stop_atr * atr[i]
                        stop = candidate if not np.isfinite(stop) else min(stop, candidate)
                        stopped = close[i] >= stop
                        regime_exit = close[i] >= ma[i]
                    if stopped or regime_exit or held >= params.hold_bars:
                        position = 0
                # Exit decisions always produce a flat next-bar target.
                signal[i] = position
            elif entry[i]:
                position = int(entry[i])
                held = 0
                extreme = np.nan
                stop = np.nan
                signal[i] = position
        df['signal'] = signal
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        size = (params.target_vol / indicators['risk_vol'].where(indicators['risk_vol'] > 0.0)).replace([np.inf, -np.inf], np.nan)
        df['size'] = size.clip(lower=0.000001, upper=params.max_size).shift(1).fillna(0.000001).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=12, freq='B')
    close = np.array([100., 102., 106., 101., 100., 100., 98., 94., 99., 100., 100., 100.])
    data = pd.DataFrame({'open': close, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'ma200': [90.] * 5 + [110.] * 7, 'atr': 1.0, 'breadth': 4.0, 'coherence': [1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.], 'risk_vol': 0.24}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, WaveletParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0, 0]
    assert np.isfinite(result['size']).all() and (result['size'] > 0).all()
    expiry = strategy.generate_signals(data, ind, None, WaveletParams(hold_bars=2)).data
    assert expiry['signal'].iloc[3] == 0 and expiry['signal'].iloc[8] == 0
    prefix = strategy.generate_signals(data.iloc[:8], ind.iloc[:8], None, WaveletParams()).data
    assert prefix.equals(result.iloc[:8])
    assert strategy.warmup_bars(WaveletParams(energy_window=250)) >= 266


if __name__ == '__main__':
    _self_check()
