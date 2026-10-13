from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 14
    smooth_window: int = 3
    strength_threshold: float = 0.45
    atr_window: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local-w2_1790983746'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(params.graph_window + params.smooth_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.graph_window < 3 or params.smooth_window < 1 or params.atr_window < 1:
            raise ValueError('Indicator windows must be positive; graph_window must be at least three.')
        if not 0.0 < params.strength_threshold < 1.0:
            raise ValueError('strength_threshold must lie between zero and one.')
        if params.breakeven_pct <= 0 or params.trail_atr <= 0 or params.initial_stop_atr <= 0:
            raise ValueError('Stop parameters must be positive.')
        if not 3 <= params.max_hold <= 5:
            raise ValueError('max_hold must be between three and five sessions.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        x = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        upper = pd.Series(0.0, index=data.index)
        lower = pd.Series(0.0, index=data.index)
        signed = pd.Series(0.0, index=data.index)
        # Horizontal visibility: intermediate prices must be below both endpoints.
        # Applying the same construction to inverted prices captures peak bridges.
        for distance in range(2, params.graph_window):
            past = x.shift(distance)
            ceiling = x.rolling(distance - 1, min_periods=distance - 1).max().shift(1)
            floor = x.rolling(distance - 1, min_periods=distance - 1).min().shift(1)
            up_edge = ceiling.lt(np.minimum(x, past)).astype(float)
            down_edge = floor.gt(np.maximum(x, past)).astype(float)
            upper += up_edge
            lower += down_edge
            signed += (up_edge + down_edge) * np.sign(x - past).fillna(0.0)
        complete = x.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        upper = upper.where(complete)
        lower = lower.where(complete)
        signed = signed.where(complete)
        bridges = upper + lower
        numerator = signed.rolling(params.smooth_window, min_periods=params.smooth_window).sum()
        denominator = bridges.rolling(params.smooth_window, min_periods=params.smooth_window).sum()
        score = numerator.div(denominator.where(denominator > 0)).clip(-1.0, 1.0)
        previous = close.shift(1)
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out['log_close'] = x
        out['upper_degree'] = upper
        out['lower_degree'] = lower
        out['signed_bridges'] = signed
        out['bridge_count'] = bridges
        out['smoothed_signed_bridges'] = numerator
        out['smoothed_bridge_count'] = denominator
        out['score'] = score
        out['true_range'] = tr
        out['atr'] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        score = indicators['score']
        confirmed_long = score.ge(params.strength_threshold) & score.shift(1).ge(params.strength_threshold)
        confirmed_short = score.le(-params.strength_threshold) & score.shift(1).le(-params.strength_threshold)
        candidates = np.select([confirmed_long, confirmed_short], [1, -1], default=0)
        strength = (0.25 + 0.70 * score.abs().fillna(0.0)).clip(0.25, 0.95).to_numpy()
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decision = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        side = 0
        entry = 0.0
        stop = 0.0
        best = 0.0
        armed = False
        age = 0
        exposure = 0.25
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            exited = False
            # Yesterday's decision fills at today's open; never inspect tomorrow.
            if side == 0 and i > 0 and decision[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    side = int(decision[i - 1])
                    entry = opens[i]
                    exposure = sizes[i - 1]
                    stop = -params.initial_stop_atr * atr[i - 1]
                    best = 0.0
                    armed = False
                    age = 0
                else:
                    exited = True
            if side != 0:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    side = 0
                    exited = True
                else:
                    favorable = side * (closes[i] - entry)
                    best = max(best, favorable)
                    if best >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        # Ratchet in favorable coordinates: long stops rise,
                        # short cover stops fall. Neither ever loosens.
                        stop = max(stop, 0.0, best - params.trail_atr * atr[i])
                    # Stops are detected at the close and executed next bar.
                    if favorable <= stop or age >= params.max_hold:
                        side = 0
                        exited = True
                decision[i] = side
                if side != 0:
                    sizes[i] = exposure
            if side == 0 and not exited and i >= warmup - 1:
                if candidates[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0:
                    if np.isfinite(closes[i]) and closes[i] > 0:
                        decision[i] = int(candidates[i])
                        sizes[i] = float(strength[i])
        df = pd.DataFrame({'signal': decision, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def demo() -> None:
    params = VisibilityParams()
    strategy = GeneratedStrategy()
    n = 80
    index = pd.date_range('2020-01-01', periods=n, freq='B')
    start = strategy.warmup_bars(params) - 1
    ctx = StrategyContext('SPY', '1d', strategy.warmup_bars(params))
    for direction in (1, -1):
        close = np.full(n, 100.0)
        close[start + 1] = 100.0 + direction * 1.0
        close[start + 2] = 100.0 + direction * 3.0
        close[start + 3] = 100.0 + direction * 0.8
        data = pd.DataFrame({'open': 100.0, 'high': np.maximum(close, 100.0) + 0.5,
                             'low': np.minimum(close, 100.0) - 0.5,
                             'close': close, 'volume': 1000000.0}, index=index)
        forced = pd.DataFrame({'score': direction * 0.8, 'atr': 1.0}, index=index)
        signals = strategy.generate_signals(data, forced, ctx, params).data
        assert signals['signal'].iloc[start] == 0
        assert signals['signal'].iloc[start + 1] == direction
        assert signals['signal'].iloc[start + 3] == direction
        assert signals['signal'].iloc[start + 4] == 0
        assert np.isclose(signals['size'].iloc[start + 1], 0.81)
        actual = strategy.indicators(data, params)
        full = strategy.generate_signals(data, actual, ctx, params).data
        prefix = data.iloc[:start + 5]
        partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
        pd.testing.assert_frame_equal(full.iloc[:len(prefix)], partial)
        assert full['signal'].isin([-1, 0, 1]).all()
        assert np.isfinite(full['size']).all() and full['size'].gt(0).all()


if __name__ == '__main__':
    demo()
