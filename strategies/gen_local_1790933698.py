from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    ma_window: int = 12
    corr_window: int = 40
    atr_window: int = 14
    z_entry: float = 1.2
    min_pull: float = 0.4
    breakeven_pct: float = 0.004
    trail_atr: float = 0.8
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = 'gen_local_1790933698'

    @classmethod
    def params_type(cls):
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        return int(max(params.ma_window + params.corr_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        if min(params.ma_window, params.corr_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two.')
        if not (params.z_entry > 0 and params.min_pull > 0):
            raise ValueError('Entry thresholds must be positive.')
        close = data['close']
        mean = close.rolling(params.ma_window).mean()
        std = close.rolling(params.ma_window).std(ddof=0).replace(0.0, np.nan)
        z = ((close - mean) / std).replace([np.inf, -np.inf], np.nan)
        lag_z = z.shift(1)
        rho = z.rolling(params.corr_window).corr(lag_z)
        rho = rho.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        pull = z.abs() * (1.0 - rho)
        previous_close = close.shift(1)
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean().replace(0.0, np.nan)
        ready = rho.between(0.0, 0.85) & pull.ge(params.min_pull) & atr.gt(0.0)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[ready & z.le(-params.z_entry)] = 1
        candidate.loc[ready & z.ge(params.z_entry)] = -1
        confirmed = candidate.ne(0) & candidate.eq(candidate.shift(1))
        entry = candidate.where(confirmed, 0).astype(int)
        return pd.DataFrame({
            'mean': mean, 'std': std, 'z': z, 'lag_z': lag_z,
            'rho': rho, 'pull': pull, 'previous_close': previous_close,
            'tr': tr, 'atr': atr, 'ready': ready,
            'candidate': candidate, 'confirmed': confirmed, 'entry': entry,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ElasticParams) -> SignalFrame:
        if not (params.trail_atr > 0 and params.breakeven_pct > 0):
            raise ValueError('Stop parameters must be positive.')
        if params.max_hold not in (1, 2):
            raise ValueError('Holding horizon must be one or two sessions.')
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        direction = 0
        age = 0
        entry_price = 0.0
        stop = 0.0
        armed = False
        for i in range(len(data)):
            if direction:
                # The previous close's decision fills at this bar's open.
                if age == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        direction = 0
                        continue
                    # Signed-price coordinates make both sides ratchet upward.
                    stop = direction * entry_price - params.trail_atr * atrs[i - 1]
                    armed = False
                age += 1
                close = closes[i]
                atr = atrs[i]
                if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                    direction = 0
                    continue
                signed_close = direction * close
                # Stops are observed at the close; exits fill next open.
                if signed_close <= stop:
                    direction = 0
                    continue
                if direction * (close / entry_price - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, direction * entry_price,
                               signed_close - params.trail_atr * atr)
                if age >= params.max_hold:
                    direction = 0
                    continue
                signals[i] = direction
            elif entries[i] in (-1, 1):
                if np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
                    direction = int(entries[i])
                    age = 0
                    signals[i] = direction
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = ElasticParams()
    index = pd.date_range('2020-01-01', periods=100)
    flat = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    derived = strategy.indicators(flat, params)
    assert derived['entry'].eq(0).all()
    assert strategy.generate_signals(flat, derived, None, params).data['signal'].eq(0).all()
    for side in (1, -1):
        sample = flat.iloc[:6].copy()
        sample['close'] = [100.0, 100.0 + side, 100.0, 100.0, 100.0, 100.0]
        inputs = pd.DataFrame({'atr': 1.0, 'entry': [side, 0, 0, 0, 0, 0]}, index=sample.index)
        result = strategy.generate_signals(sample, inputs, None, params).data
        assert result['signal'].tolist() == [0, side, side, 0, 0, 0]
        assert result['size'].gt(0).all()
