from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalDrawdownParams:
    drawdown_window: int = 63
    season_window: int = 504
    atr_window: int = 14
    stress_depth: float = 0.02
    min_samples: int = 8
    edge_threshold: float = 0.35
    holding_bars: int = 18
    breakeven_gain: float = 0.02
    trail_atr: float = 2.5
    min_size: float = 0.15
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.drawdown_window, self.season_window, self.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two.')
        if not 2 <= self.min_samples <= self.season_window:
            raise ValueError('Invalid minimum sample count.')
        if not 0 < self.stress_depth < 1 or self.edge_threshold <= 0:
            raise ValueError('Invalid drawdown or seasonal threshold.')
        if self.holding_bars < 1 or self.breakeven_gain <= 0 or self.trail_atr <= 0:
            raise ValueError('Invalid exit parameters.')
        if not 0 < self.min_size <= self.max_size <= 1:
            raise ValueError('Sizes must lie in (0, 1].')


class GeneratedStrategy(BaseStrategy[SeasonalDrawdownParams]):
    strategy_id = 'gen_local_1791312839'

    @classmethod
    def params_type(cls):
        return SeasonalDrawdownParams

    def warmup_bars(self, params):
        return int(max(params.drawdown_window + params.season_window + 1,
                       params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        peak = close.rolling(params.drawdown_window,
                             min_periods=params.drawdown_window).max()
        out['peak'] = peak
        out['drawdown'] = (1.0 - close / peak.where(peak > 0)).clip(lower=0)
        out['previous_drawdown'] = out['drawdown'].shift(1)
        out['return'] = close.pct_change(fill_method=None)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        month = data.index.month
        stressed = out['previous_drawdown'].ge(params.stress_depth)
        valid_return = out['return'].replace([np.inf, -np.inf], np.nan)
        for name in ('season_count', 'season_mean', 'season_std', 'season_score'):
            out[name] = 0.0
        for m in range(1, 13):
            eligible = stressed & (month == m) & valid_return.notna()
            observations = eligible.astype(float)
            values = valid_return.where(eligible, 0.0)
            count = observations.rolling(
                params.season_window, min_periods=params.season_window).sum().shift(1)
            total = values.rolling(
                params.season_window, min_periods=params.season_window).sum().shift(1)
            squares = values.pow(2).rolling(
                params.season_window, min_periods=params.season_window).sum().shift(1)
            mean = total / count.where(count > 0)
            variance = ((squares - total.pow(2) / count.where(count > 0)) /
                        (count - 1).where(count > 1)).clip(lower=0)
            std = np.sqrt(variance)
            score = mean * np.sqrt(count) / std.where(std > 1e-12)
            score = score.where(count >= params.min_samples, 0.0)
            mask = month == m
            for name, series in (('season_count', count), ('season_mean', mean),
                                 ('season_std', std), ('season_score', score)):
                out.loc[mask, name] = series.loc[mask].replace(
                    [np.inf, -np.inf], np.nan).fillna(0.0)
        out['strength'] = (
            out['season_score'].abs() / (out['season_score'].abs() + 1.0) *
            (out['drawdown'] / (3.0 * params.stress_depth)).clip(0.0, 1.0)
        ).fillna(0.0)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        high = data['high'].to_numpy(dtype=float)
        low = data['low'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        depth = indicators['drawdown'].to_numpy(dtype=float)
        score = indicators['season_score'].to_numpy(dtype=float)
        strength = indicators['strength'].to_numpy(dtype=float)
        direction = 0
        entry = np.nan
        stop = -np.inf
        armed = False
        age = 0
        position_size = params.min_size
        ready = self.warmup_bars(params) - 1
        for i in range(n):
            valid_bar = (np.isfinite([opening[i], high[i], low[i], close[i], atr[i]]).all()
                         and min(opening[i], high[i], low[i], close[i], atr[i]) > 0)
            if direction:
                if not valid_bar:
                    direction = 0
                    entry = np.nan
                    continue
                if np.isnan(entry):
                    # The previous close's order fills at this bar's open.
                    entry = opening[i]
                    stop = direction * entry - params.trail_atr * atr[i - 1]
                age += 1
                adverse = low[i] if direction == 1 else -high[i]
                if adverse <= stop or age >= params.holding_bars:
                    # Exit decision executes next bar, like every other signal.
                    direction = 0
                    entry = np.nan
                    continue
                gain = direction * (close[i] / entry - 1.0)
                armed = armed or gain >= params.breakeven_gain
                if armed:
                    # Signed coordinates make both long and short stops ratchet upward.
                    stop = max(stop, direction * entry,
                               direction * close[i] - params.trail_atr * atr[i])
            elif (i >= ready and valid_bar and np.isfinite(depth[i])
                  and depth[i] >= params.stress_depth
                  and np.isfinite(score[i]) and abs(score[i]) >= params.edge_threshold):
                direction = 1 if score[i] > 0 else -1
                entry = np.nan
                armed = False
                age = 0
                position_size = float(np.clip(
                    params.min_size + (params.max_size - params.min_size) * strength[i],
                    params.min_size, params.max_size))
            if direction:
                signals[i] = direction
                sizes[i] = position_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = SeasonalDrawdownParams(drawdown_window=2, season_window=2,
                                    atr_window=2, min_samples=2, trail_atr=2.0)
    index = pd.date_range('2020-01-01', periods=10)
    for direction in (1, -1):
        prices = np.full(10, 100.0)
        prices[6] = 100.0 + direction * 3.0
        data = pd.DataFrame({'open': 100.0, 'high': prices + 0.2,
                             'low': prices - 0.2, 'close': prices,
                             'volume': 1000.0}, index=index)
        ind = pd.DataFrame({'atr': 1.0, 'drawdown': 0.05,
                            'season_score': 0.0, 'strength': 0.8}, index=index)
        ind.loc[index[4], 'season_score'] = direction
        result = GeneratedStrategy.generate_signals(
            GeneratedStrategy, data, ind, None, params).data
        assert result['signal'].tolist() == [0, 0, 0, 0, 0, direction, direction, direction, 0, 0]
        assert result.loc[index[5], 'size'] > params.min_size
        assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
