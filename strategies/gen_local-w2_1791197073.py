from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_bars: int = 12
    history_bars: int = 126
    compression_q: float = 0.25
    density_q: float = 0.75
    volume_q: float = 0.80
    profit_target: float = 0.05
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791197073'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.range_bars, 20) + params.history_bars + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.range_bars < 2 or params.history_bars < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not all(0.0 < q < 1.0 for q in (
            params.compression_q, params.density_q, params.volume_q
        )):
            raise ValueError('Percentiles must lie strictly between zero and one.')
        if params.profit_target <= 0.0 or params.max_hold_bars < 1:
            raise ValueError('Profit target and holding period must be positive.')

        close = data['close'].where(data['close'] > 0.0)
        volume = data['volume'].where(data['volume'] >= 0.0)
        upper = data['high'].rolling(params.range_bars).max()
        lower = data['low'].rolling(params.range_bars).min()
        width = ((upper - lower) / close).where(upper > lower)
        baseline_volume = volume.rolling(20).mean().replace(0.0, np.nan)
        density = volume.rolling(params.range_bars).sum() / baseline_volume / width
        density = density.replace([np.inf, -np.inf], np.nan)

        out = pd.DataFrame(index=data.index)
        out['upper'] = upper.shift(1)
        out['lower'] = lower.shift(1)
        out['width'] = width.shift(1)
        out['density'] = density.shift(1)
        # Threshold histories exclude the candidate compression window's last bar.
        out['width_threshold'] = width.rolling(params.history_bars).quantile(
            params.compression_q
        ).shift(2)
        out['density_threshold'] = density.rolling(params.history_bars).quantile(
            params.density_q
        ).shift(2)
        out['volume_threshold'] = volume.rolling(params.history_bars).quantile(
            params.volume_q
        ).shift(1)

        ready = out.notna().all(axis=1)
        congested = (
            (out['width'] <= out['width_threshold'])
            & (out['density'] >= out['density_threshold'])
        )
        confirmed = volume > out['volume_threshold']
        long_entry = (
            ready & congested & confirmed
            & (close > out['upper']) & (close > data['open'])
        )
        short_entry = (
            ready & congested & confirmed
            & (close < out['lower']) & (close < data['open'])
        )
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if params.profit_target <= 0.0 or params.max_hold_bars < 1:
            raise ValueError('Profit target and holding period must be positive.')
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        held_bars = 0

        for i in range(len(data)):
            if position:
                # A previous close's entry decision fills at this bar's open.
                if held_bars == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        position = 0
                        continue
                held_bars += 1
                gain = position * (closes[i] / entry_price - 1.0)
                if (
                    (np.isfinite(gain) and gain >= params.profit_target)
                    or held_bars >= params.max_hold_bars
                ):
                    position = 0
                    entry_price = np.nan
                    held_bars = 0
                decisions[i] = position
            elif entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(entries[i])
                held_bars = 0
                decisions[i] = position

        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    sample = pd.DataFrame({
        'open': [90.0, 100.0, 101.0, 106.0, 100.0, 99.0],
        'high': [101.0, 102.0, 107.0, 107.0, 101.0, 100.0],
        'low': [89.0, 99.0, 100.0, 105.0, 98.0, 98.0],
        'close': [100.0, 101.0, 106.0, 106.0, 99.0, 99.0],
        'volume': [100.0] * 6,
    }, index=index)
    entries = pd.DataFrame({'entry': [1, 0, 0, -1, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(
        sample, entries, None, Params(profit_target=0.05, max_hold_bars=2)
    )
    assert result.data['signal'].tolist() == [0, 1, 1, 0, -1, -1]
    assert result.data['size'].gt(0.0).all()
    falling = sample.copy()
    falling['close'] = [100.0, 99.0, 94.0, 94.0, 99.0, 99.0]
    short_entries = pd.DataFrame({'entry': [-1, 0, 0, 0, 0, 0]}, index=index)
    short_result = GeneratedStrategy.generate_signals(
        falling, short_entries, None, Params(max_hold_bars=20)
    )
    assert short_result.data['signal'].tolist() == [0, -1, -1, 0, 0, 0]
