from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 12
    noise_window: int = 20
    floor_window: int = 4
    reference_window: int = 8
    atr_window: int = 14
    yield_threshold: float = 1.5
    recoil_limit: float = 1.25
    stop_atr: float = 2.0
    hold_bars: int = 8
    min_size: float = 0.2
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1790940354'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(
            params.atr_window + 1,
            params.slope_window + params.noise_window + 1,
            params.slope_window + params.floor_window + params.reference_window,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.slope_window < 3 or params.noise_window < 2
                or params.floor_window < 1 or params.reference_window < 1
                or params.atr_window < 1 or params.yield_threshold <= 0
                or params.recoil_limit <= 0):
            raise ValueError('Invalid indicator parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_close'] = np.log(close.where(close > 0))
        clock = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        window = params.slope_window
        out['log_sum'] = out['log_close'].rolling(window).sum()
        out['weighted_log_sum'] = (clock * out['log_close']).rolling(window).sum()
        out['slope'] = (
            out['weighted_log_sum']
            - (clock - (window - 1) / 2.0) * out['log_sum']
        ) / (window * (window * window - 1) / 12.0)
        out['innovation'] = out['slope'].diff()
        out['innovation_noise'] = out['innovation'].rolling(params.noise_window).std(ddof=0)
        noise = out['innovation_noise'].where(out['innovation_noise'] > 1e-12)
        out['innovation_z'] = out['innovation'] / noise
        out['slope_floor'] = out['slope'].rolling(params.floor_window).min()
        out['previous_floor'] = out['slope_floor'].shift(params.reference_window)
        out['plastic_shift'] = (out['slope_floor'] - out['previous_floor']) / noise
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        uplift = out['plastic_shift'].clip(lower=0)
        recoil = (1.0 - out['innovation_z'].abs() / params.recoil_limit).clip(0.0, 1.0)
        out['strength'] = (uplift / (uplift + params.yield_threshold)) * recoil
        out = out.replace([np.inf, -np.inf], np.nan)
        out['entry'] = (
            (out['slope_floor'] > 0)
            & (out['plastic_shift'] >= params.yield_threshold)
            & (out['innovation_z'] < 0)
            & (out['innovation_z'] > -params.recoil_limit)
            & (out['atr'] > 0)
            & out['strength'].notna()
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if (params.stop_atr <= 0 or params.hold_bars < 1
                or not 0 < params.min_size <= params.max_size <= 1):
            raise ValueError('Invalid execution parameters')
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['strength'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        active = False
        pending = False
        stop = np.nan
        entry_atr = 0.0
        age = 0
        exposure = 1.0
        for i in range(len(data)):
            if active:
                if pending:
                    # The preceding close scheduled entry at this bar's open.
                    # Freeze the preceding ATR and actual entry-bar open.
                    stop = opens[i] - params.stop_atr * entry_atr
                    pending = False
                age += 1
                if (not np.isfinite(closes[i]) or not np.isfinite(stop)
                        or closes[i] < stop or age >= params.hold_bars):
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = exposure
                continue
            if (entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0
                    and np.isfinite(strengths[i])):
                active = True
                pending = True
                age = 0
                entry_atr = atrs[i]
                exposure = params.min_size + (params.max_size - params.min_size) * float(
                    np.clip(strengths[i], 0.0, 1.0)
                )
                signals[i] = 1
                sizes[i] = exposure
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    forced = pd.DataFrame({
        'atr': 1.0, 'entry': [True] + [False] * 11, 'strength': 0.5,
    }, index=index)
    result = strategy.generate_signals(data, forced, None, params).data
    assert result['signal'].tolist() == [0] + [1] * 8 + [0] * 3
    assert np.isclose(result['size'].iloc[1], 0.6)
    # A rally cannot raise the frozen stop at 98.
    data.loc[index[1:4], 'close'] = [120.0, 99.0, 97.0]
    result = strategy.generate_signals(data, forced, None, params).data
    assert result['signal'].iloc[:5].tolist() == [0, 1, 1, 1, 0]
    # An entry gap changes the stop anchor to the entry open.
    data.loc[index[1], ['open', 'close']] = [110.0, 105.0]
    result = strategy.generate_signals(data, forced, None, params).data
    assert result['signal'].iloc[:3].tolist() == [0, 1, 0]
    empty_entries = strategy.indicators(data, params)
    assert not empty_entries['entry'].any()
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
