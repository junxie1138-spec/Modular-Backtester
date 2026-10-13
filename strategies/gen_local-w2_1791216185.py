from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionShiftParams:
    rank_window: int = 63
    recent_window: int = 5
    baseline_window: int = 20
    change_window: int = 126
    change_percentile: float = 0.9
    pressure_threshold: float = 0.65
    atr_window: int = 14
    initial_stop_atr: float = 3.0
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    hold_bars: int = 18

    def __post_init__(self):
        windows = (self.rank_window, self.recent_window, self.baseline_window, self.change_window, self.atr_window)
        if any(w < 2 for w in windows) or self.hold_bars < 1:
            raise ValueError('Lookbacks must be at least two and hold_bars positive.')
        if not 0.5 < self.pressure_threshold < 1.0:
            raise ValueError('pressure_threshold must be between 0.5 and 1.')
        if not 0.0 < self.change_percentile <= 1.0:
            raise ValueError('change_percentile must be between 0 and 1.')
        risk = (self.initial_stop_atr, self.breakeven_pct, self.trail_atr)
        if any(not np.isfinite(v) or v <= 0.0 for v in risk):
            raise ValueError('Stop and activation parameters must be finite and positive.')


class GeneratedStrategy(BaseStrategy[AuctionShiftParams]):
    strategy_id = 'gen_local-w2_1791216185'

    @classmethod
    def params_type(cls):
        return AuctionShiftParams

    @staticmethod
    def warmup_bars(params: AuctionShiftParams) -> int:
        return int(max(200, params.atr_window + 1, params.rank_window + params.baseline_window + params.recent_window + params.change_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AuctionShiftParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0.0)
        span = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['closing_pressure'] = ((2.0 * close - data['high'] - data['low']) / span).clip(-1.0, 1.0)
        out['pressure_rank'] = out['closing_pressure'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['recent_pressure'] = out['pressure_rank'].rolling(params.recent_window, min_periods=params.recent_window).mean()
        out['baseline_pressure'] = out['pressure_rank'].rolling(params.baseline_window, min_periods=params.baseline_window).mean().shift(params.recent_window)
        out['pressure_change'] = out['recent_pressure'] - out['baseline_pressure']
        out['change_magnitude'] = out['pressure_change'].abs()
        out['change_rank'] = out['change_magnitude'].rolling(params.change_window, min_periods=params.change_window).rank(pct=True)
        previous_close = close.shift(1)
        components = pd.concat([span, (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1)
        out['true_range'] = components.max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AuctionShiftParams) -> SignalFrame:
        extreme = indicators['change_rank'].ge(params.change_percentile)
        fresh = extreme & ~extreme.shift(1, fill_value=False)
        bullish = fresh & indicators['recent_pressure'].ge(params.pressure_threshold) & indicators['pressure_change'].gt(0.0) & data['close'].gt(indicators['ma200'])
        bearish = fresh & indicators['recent_pressure'].le(1.0 - params.pressure_threshold) & indicators['pressure_change'].lt(0.0) & data['close'].lt(indicators['ma200'])
        candidates = np.where(bullish, 1, np.where(bearish, -1, 0))
        required = indicators[['recent_pressure', 'pressure_change', 'change_rank', 'atr', 'ma200']].to_numpy(dtype=float)
        ready = np.isfinite(required).all(axis=1)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        averages = indicators['ma200'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        stop = np.nan
        best = np.nan
        armed = False
        age = 0

        for i in range(len(data)):
            if not ready[i] or not np.isfinite(closes[i]) or closes[i] <= 0.0 or atrs[i] <= 0.0:
                position = 0
                entry = np.nan
                continue
            if position:
                # The previous close's decision fills at this session's open.
                if np.isnan(entry):
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        position = 0
                        continue
                    entry = opens[i]
                    best = position * entry
                    stop = best - params.initial_stop_atr * atrs[i - 1]
                    armed = False
                    age = 0
                age += 1
                signed_close = position * closes[i]
                best = max(best, signed_close)
                favorable_return = position * (closes[i] / entry - 1.0)
                if favorable_return >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed-price stops only rise; a short's actual buy-stop falls.
                    stop = max(stop, position * entry, best - params.trail_atr * atrs[i])
                regime_broken = position * (closes[i] - averages[i]) <= 0.0
                # Stops are evaluated at the close and exit at the next open.
                if signed_close <= stop or regime_broken or age >= params.hold_bars:
                    position = 0
                    entry = np.nan
            elif candidates[i]:
                position = int(candidates[i])
                entry = np.nan
            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    p = AuctionShiftParams()
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    for side in (1, -1):
        closes = np.array([100.0, 101.0, 105.0, 101.0, 101.0, 101.0])
        if side == -1:
            closes = 200.0 - closes
        data = pd.DataFrame({'open': closes, 'high': closes + 1.0, 'low': closes - 1.0, 'close': closes, 'volume': 1000.0}, index=index)
        ind = pd.DataFrame({'recent_pressure': 0.8 if side == 1 else 0.2, 'pressure_change': side * 0.3, 'change_rank': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'atr': 1.0, 'ma200': 90.0 if side == 1 else 110.0}, index=index)
        result = GeneratedStrategy.generate_signals(data, ind, None, p).data
        assert result['signal'].tolist() == [0, side, side, side, 0, 0]
        assert result['size'].gt(0.0).all()
    index = pd.date_range('2020-01-01', periods=300, freq='D')
    t = np.arange(300, dtype=float)
    close = 100.0 + 0.04 * t + 2.0 * np.sin(t / 7.0)
    data = pd.DataFrame({'open': close, 'high': close + 1.5, 'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
    full = GeneratedStrategy.indicators(data, p)
    prefix = GeneratedStrategy.indicators(data.iloc[:260], p)
    pd.testing.assert_frame_equal(full.iloc[:260], prefix)
    full_signal = GeneratedStrategy.generate_signals(data, full, None, p).data
    prefix_signal = GeneratedStrategy.generate_signals(data.iloc[:260], prefix, None, p).data
    pd.testing.assert_frame_equal(full_signal.iloc[:260], prefix_signal)
    assert full.dropna().index[0] <= index[GeneratedStrategy.warmup_bars(p) - 1]
