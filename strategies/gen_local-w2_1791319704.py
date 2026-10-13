from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RepairParams:
    ma_window: int = 20
    atr_window: int = 14
    peak_window: int = 10
    discount_z: float = 0.5
    repair_ratio: float = 1.5
    drawdown_atr: float = 1.0
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RepairParams]):
    strategy_id = 'gen_local-w2_1791319704'

    @classmethod
    def params_type(cls) -> type[RepairParams]:
        return RepairParams

    def warmup_bars(self, params: RepairParams) -> int:
        if min(params.ma_window, params.atr_window, params.peak_window) < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not 3 <= params.hold_bars <= 5:
            raise ValueError('Holding horizon must be three to five sessions.')
        thresholds = [params.discount_z, params.repair_ratio, params.drawdown_atr, params.stop_atr]
        if not all(np.isfinite(x) and x > 0 for x in thresholds):
            raise ValueError('Thresholds must be finite and positive.')
        return int(max(params.ma_window + 2, params.atr_window + 1, params.peak_window + 1))

    def indicators(self, data: pd.DataFrame, params: RepairParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data['close']
        out = pd.DataFrame(index=data.index)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0).replace(0.0, np.nan)
        out['z'] = (close - out['ma']) / out['std']
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean().replace(0.0, np.nan)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown_atr'] = (out['peak'] - close) / out['atr']
        out['price_repair'] = close.diff() / out['std'].shift(1)
        out['ma_drift'] = out['ma'].diff().abs() / out['std'].shift(1)
        out['z_change'] = out['z'].diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['repair'] = (
            (out['z'] <= -params.discount_z)
            & (out['z_change'] > 0.0)
            & (out['price_repair'] > params.repair_ratio * out['ma_drift'])
            & (out['drawdown_atr'] >= params.drawdown_atr)
            & (out['atr'] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RepairParams) -> SignalFrame:
        self.warmup_bars(params)
        repair = indicators['repair'].fillna(False).astype(bool)
        confirmed = (repair & repair.shift(1, fill_value=False)).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        pending = False
        entry_bar = None
        stop = np.nan

        for i in range(len(data)):
            if pending:
                pending = False
                if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0.0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0.0:
                    continue
                entry_bar = i
                # Freeze the stop at the entry open using the decision bar's ATR.
                stop = opens[i] - params.stop_atr * atr[i - 1]

            if entry_bar is not None:
                if not np.isfinite(closes[i]) or closes[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    entry_bar = None
                    continue
                raw[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0.0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                raw[i] = 1
                pending = True

        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=8, freq='B')
    data = pd.DataFrame({
        'open': 100.0, 'high': 105.0, 'low': 96.0,
        'close': [100.0, 100.0, 100.0, 97.0, 100.0, 100.0, 100.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        'repair': [True, True, False, False, False, False, False, False],
        'atr': [1.0, 1.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    }, index=index)
    strategy = GeneratedStrategy()
    params = RepairParams()
    stopped = strategy.generate_signals(data, ind, None, params).data
    assert stopped['signal'].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    data['close'] = [100.0, 100.0, 100.0, 104.0, 99.0, 100.0, 100.0, 100.0]
    timed = strategy.generate_signals(data, ind, None, params).data
    assert timed['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert (timed['size'] > 0.0).all()
    flat = data.copy()
    flat[['open', 'high', 'low', 'close']] = 100.0
    assert not strategy.indicators(flat, params)['repair'].any()


if __name__ == '__main__':
    _self_check()
