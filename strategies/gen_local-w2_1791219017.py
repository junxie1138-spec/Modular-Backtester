from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    slow_energy_min: float = 0.55
    correlation_min: float = 0.15
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local-w2_1791219017'

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    @staticmethod
    def warmup_bars(params: WaveletParams) -> int:
        # MA: 200; longest nested return calculation: 92 observations.
        return 200

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if not (0.0 <= params.slow_energy_min <= 1.0):
            raise ValueError('slow_energy_min must be between zero and one')
        if not (0.0 <= params.correlation_min < 1.0):
            raise ValueError('correlation_min must be between zero and one')
        if not (params.breakeven_pct > 0 and params.trail_atr > 0):
            raise ValueError('Stop parameters must be positive')
        if params.max_hold_bars < 1:
            raise ValueError('max_hold_bars must be positive')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['return'] = np.log(close.where(close > 0)).diff()
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        for scale in (4, 16, 32):
            half = scale // 2
            block = out['return'].rolling(half, min_periods=half).mean()
            out[f'block_{scale}'] = block
            # Causal Haar contrast of adjacent blocks, normalized by scale.
            detail = np.sqrt(half / 2.0) * (block - block.shift(half))
            out[f'haar_{scale}'] = detail
            out[f'energy_{scale}'] = detail.pow(2).rolling(20, min_periods=20).mean()
        total = out[['energy_4', 'energy_16', 'energy_32']].sum(axis=1, min_count=3)
        out['total_energy'] = total
        out['slow_energy_share'] = (out['energy_16'] + out['energy_32']) / total.where(total > 0)
        # Sixteen-bar displacement blocks do not overlap with their lagged partner.
        block = out['block_32']
        out['block_correlation'] = block.rolling(60, min_periods=60).corr(block.shift(16))
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        required = ['ma200', 'slow_energy_share', 'block_correlation', 'block_32', 'haar_32', 'atr']
        ready = indicators[required].notna().all(axis=1)
        persistent = (
            ready
            & (indicators['slow_energy_share'] >= params.slow_energy_min)
            & (indicators['block_correlation'] >= params.correlation_min)
            & (indicators['atr'] > 0)
        )
        long_entry = (
            persistent
            & (data['close'] > indicators['ma200'])
            & (indicators['block_32'] > 0)
            & (indicators['haar_32'] > 0)
        )
        short_entry = (
            persistent
            & (data['close'] < indicators['ma200'])
            & (indicators['block_32'] < 0)
            & (indicators['haar_32'] < 0)
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = 0
        stop = -np.inf
        armed = False
        for i in range(len(data)):
            # Yesterday's target becomes today's position at today's open.
            target = int(desired[i - 1]) if i else 0
            if target != position:
                position = target
                armed = False
                if position:
                    entry_price = opens[i]
                    entry_bar = i
                    previous_atr = atrs[i - 1]
                    if not (np.isfinite(entry_price) and entry_price > 0 and np.isfinite(previous_atr) and previous_atr > 0):
                        position = 0
                        desired[i] = 0
                        continue
                    # In signed-price coordinates both long and short stops
                    # ratchet upward; a short's equivalent market stop declines.
                    stop = position * entry_price - params.trail_atr * previous_atr
            if position:
                price = closes[i]
                if not (np.isfinite(price) and price > 0 and np.isfinite(atrs[i]) and atrs[i] > 0):
                    desired[i] = 0
                    continue
                signed_price = position * price
                stopped = signed_price <= stop
                favorable_return = position * (price / entry_price - 1.0)
                if favorable_return >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, position * entry_price, signed_price - params.trail_atr * atrs[i])
                stopped = stopped or signed_price <= stop
                expired = i - entry_bar + 1 >= params.max_hold_bars
                # Close-observed exits execute at the following open.
                desired[i] = 0 if stopped or expired else position
            elif i >= GeneratedStrategy.warmup_bars(params) - 1:
                desired[i] = int(candidates[i])
        df = pd.DataFrame({'signal': desired, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = WaveletParams()
    index = pd.date_range('2020-01-01', periods=210, freq='B')
    for direction in (1, -1):
        close = np.full(210, 100.0)
        close[200] = 100.0 + direction * 3.0
        close[201:] = 100.0 + direction * 1.0
        data = pd.DataFrame({'open': 100.0, 'high': close + 0.2, 'low': close - 0.2, 'close': close, 'volume': 1000.0}, index=index)
        ind = pd.DataFrame({
            'ma200': 100.0 - direction,
            'slow_energy_share': 0.8,
            'block_correlation': 0.5,
            'block_32': float(direction),
            'haar_32': float(direction),
            'atr': 0.2,
        }, index=index)
        ind.loc[index != index[199], 'slow_energy_share'] = 0.0
        signals = GeneratedStrategy.generate_signals(data, ind, None, params).data['signal']
        assert (signals.iloc[:200] == 0).all()
        assert signals.iloc[200] == direction
        assert signals.iloc[201] == direction
        assert (signals.iloc[202:] == 0).all()


if __name__ == '__main__':
    _self_check()
