from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilitySeasonParams:
    graph_window: int = 16
    training_bars: int = 504
    hold_bars: int = 8
    atr_window: int = 14
    stop_atr: float = 2.5
    entry_score: float = 0.15
    min_samples: int = 8


class GeneratedStrategy(BaseStrategy[VisibilitySeasonParams]):
    strategy_id = 'gen_local-w2_1791219304'

    @classmethod
    def params_type(cls):
        return VisibilitySeasonParams

    def warmup_bars(self, params: VisibilitySeasonParams) -> int:
        return int(max(params.training_bars + params.hold_bars + params.graph_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilitySeasonParams) -> pd.DataFrame:
        if min(params.graph_window, params.training_bars, params.hold_bars, params.atr_window, params.min_samples) < 2:
            raise ValueError('Window and sample parameters must be at least two.')
        if params.stop_atr <= 0 or params.entry_score <= 0:
            raise ValueError('Stop distance and entry threshold must be positive.')
        close = data['close'].astype(float)
        price = np.log(close.where(close > 0))
        upper = pd.Series(0.0, index=data.index)
        lower = upper.copy()
        maximum = pd.Series(-np.inf, index=data.index)
        minimum = pd.Series(np.inf, index=data.index)
        # An endpoint sees a past vertex when its slope exceeds every intervening slope.
        for lag in range(1, params.graph_window):
            slope = (price.shift(lag) - price) / lag
            upper += (slope > maximum).astype(float)
            lower += (slope < minimum).astype(float)
            maximum = pd.Series(np.maximum(maximum.to_numpy(), slope.to_numpy()), index=data.index)
            minimum = pd.Series(np.minimum(minimum.to_numpy(), slope.to_numpy()), index=data.index)
        complete = price.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        balance = ((upper - lower) / (upper + lower).replace(0, np.nan)).where(complete)
        state = pd.Series(np.select([balance > 0.15, balance < -0.15], [1.0, -1.0], default=0.0), index=data.index).where(complete)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        realized = close.pct_change(params.hold_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        origin_state = state.shift(params.hold_bars)
        origin_weekday = weekday.shift(params.hold_bars)
        mean = pd.Series(np.nan, index=data.index)
        deviation = mean.copy()
        samples = pd.Series(0.0, index=data.index)
        for day in range(5):
            for topology in (-1, 0, 1):
                observed = origin_weekday.eq(day) & origin_state.eq(topology) & realized.notna()
                count = observed.astype(float).rolling(params.training_bars, min_periods=params.training_bars).sum()
                payoff = realized.where(observed, 0.0)
                denominator = count.replace(0, np.nan)
                conditional_mean = payoff.rolling(params.training_bars, min_periods=params.training_bars).sum() / denominator
                second_moment = payoff.pow(2).rolling(params.training_bars, min_periods=params.training_bars).sum() / denominator
                conditional_deviation = (second_moment - conditional_mean.pow(2)).clip(lower=0).pow(0.5)
                current = weekday.eq(day) & state.eq(topology)
                mean.loc[current] = conditional_mean.loc[current]
                deviation.loc[current] = conditional_deviation.loc[current]
                samples.loc[current] = count.loc[current]
        score = (mean / deviation.where(deviation > 1e-8)) * samples / (samples + params.min_samples)
        score = score.where(samples >= params.min_samples).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        previous = close.shift(1)
        true_range = pd.concat([data['high'] - data['low'], (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return pd.DataFrame({'upper_degree': upper.where(complete), 'lower_degree': lower.where(complete), 'graph_balance': balance, 'topology_state': state, 'weekday': weekday, 'origin_state': origin_state, 'origin_weekday': origin_weekday, 'realized_return': realized, 'conditional_mean': mean, 'conditional_deviation': deviation, 'samples': samples, 'score': score, 'true_range': true_range, 'atr': atr}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilitySeasonParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        position = 0
        age = 0
        exposure = 0.25
        high = low = np.nan
        floor, ceiling = -np.inf, np.inf
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        scores = indicators['score'].to_numpy(dtype=float)
        for i in range(len(data)):
            close, atr, score = closes[i], atrs[i], scores[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if position:
                age += 1
                if not valid:
                    position = 0
                else:
                    # Start the watermark on the fill bar, after the entry decision.
                    high = max(high, close) if np.isfinite(high) else close
                    low = min(low, close) if np.isfinite(low) else close
                    floor = max(floor, high - params.stop_atr * atr)
                    ceiling = min(ceiling, low + params.stop_atr * atr)
                    # Highest-close drawdown exit applies to both directions.
                    # Shorts also carry a symmetric ratcheting protective exit.
                    if close < floor or (position < 0 and close > ceiling) or age >= params.hold_bars:
                        position = 0
            elif valid and np.isfinite(score) and abs(score) >= params.entry_score:
                position = 1 if score > 0 else -1
                exposure = float(np.clip(0.25 + 0.75 * abs(score), 0.25, 1.0))
                age = 0
                high = low = np.nan
                floor, ceiling = -np.inf, np.inf
            signals[i] = position
            sizes[i] = exposure if position else 0.25
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = VisibilitySeasonParams()
    index = pd.date_range('2020-01-01', periods=5, freq='B')
    for prices, direction in [([100, 102, 104, 100, 99], 1), ([100, 99, 98, 102, 103], -1)]:
        data = pd.DataFrame({'open': prices, 'high': np.array(prices) + 1, 'low': np.array(prices) - 1, 'close': prices, 'volume': 1000}, index=index)
        derived = pd.DataFrame({'atr': 1.0, 'score': [direction, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, 0]
        assert result['size'].gt(0).all()
        assert result['size'].iloc[1] == 1.0


if __name__ == '__main__':
    _self_check()
