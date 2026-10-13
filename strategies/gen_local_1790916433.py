from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    graph_span: int = 12
    memory: int = 80
    entry_strength: float = 0.18
    atr_period: int = 14
    stop_atr: float = 2.0
    hold_days: int = 4
    min_size: float = 0.25
    max_size: float = 1.0
    full_strength: float = 0.75

    def __post_init__(self):
        if not 3 <= self.hold_days <= 5:
            raise ValueError('hold_days must be between three and five')
        if self.graph_span < self.hold_days or self.memory < 3 or self.atr_period < 2:
            raise ValueError('Invalid lookback lengths')
        if not 0.0 < self.entry_strength < self.full_strength:
            raise ValueError('Invalid forecast thresholds')
        if not 0.0 < self.min_size <= self.max_size <= 1.0:
            raise ValueError('Invalid exposure bounds')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0.0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1790916433'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(
            params.hold_days + params.graph_span + params.memory + 1,
            params.atr_period + 1,
        )

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['log_close'] = np.log(close.where(close > 0.0))
        out['block_return'] = out['log_close'].diff(params.hold_days)
        out['block_std'] = out['block_return'].rolling(params.memory).std()
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan)
        out['atr'] = out['true_range'].rolling(params.atr_period).mean()
        numerator = pd.Series(0.0, index=data.index)
        denominator = pd.Series(0.0, index=data.index)
        prices = out['log_close']
        blocks = out['block_return']
        scale = out['block_std'].where(out['block_std'] > 1e-12)

        # Vectorized across bars; each edge connects only to an earlier close.
        for lag in range(params.hold_days, params.graph_span + 1):
            past = prices.shift(lag)
            upper = prices.shift(1).rolling(lag - 1).max()
            lower = prices.shift(1).rolling(lag - 1).min()
            visible_upper = (prices > upper) & (past > upper)
            visible_lower = (prices < lower) & (past < lower)
            visible = visible_upper | visible_lower
            rho = blocks.rolling(params.memory).corr(blocks.shift(lag))
            rho = rho.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
            # At t+h, an autoregressive lag L uses the already-known block t+h-L.
            predictor = blocks.shift(lag - params.hold_days)
            forecast = (rho * predictor / scale).replace([np.inf, -np.inf], np.nan)
            weight = visible.astype(float) * (lag / params.graph_span)
            weight = weight.where(forecast.notna(), 0.0)
            out[f'past_{lag}'] = past
            out[f'upper_blocker_{lag}'] = upper
            out[f'lower_blocker_{lag}'] = lower
            out[f'upper_edge_{lag}'] = visible_upper.astype(int)
            out[f'lower_edge_{lag}'] = visible_lower.astype(int)
            out[f'autocorrelation_{lag}'] = rho
            out[f'predictor_{lag}'] = predictor
            out[f'forecast_{lag}'] = forecast
            out[f'weight_{lag}'] = weight
            numerator += weight * forecast.fillna(0.0)
            denominator += weight

        out['forecast_sum'] = numerator
        out['edge_weight_sum'] = denominator
        out['score'] = (numerator / denominator.where(denominator > 0.0)).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.full(n, params.min_size, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        scores = indicators['score'].to_numpy(dtype=float)
        side = 0
        entered = -1
        exposure = params.min_size
        highest = -np.inf
        lowest = np.inf
        floor = -np.inf
        ceiling = np.inf
        first = self.warmup_bars(params) - 1

        for i in range(n):
            close, atr, score = closes[i], atrs[i], scores[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if side:
                # The first tracked close belongs to the bar after the entry decision.
                if valid:
                    highest = max(highest, close)
                    lowest = min(lowest, close)
                    floor = max(floor, highest - params.stop_atr * atr)
                    ceiling = min(ceiling, lowest + params.stop_atr * atr)
                # Both directions obey the mandated highest-close drawdown exit.
                # Shorts additionally exit on an adverse rebound from their lowest close.
                stopped = valid and (close <= floor or (side < 0 and close >= ceiling))
                if not valid or stopped or i - entered >= params.hold_days:
                    side = 0
                    continue
            elif i >= first and valid and np.isfinite(score) and abs(score) >= params.entry_strength:
                side = 1 if score > 0.0 else -1
                entered = i
                strength = np.clip(
                    (abs(score) - params.entry_strength)
                    / (params.full_strength - params.entry_strength),
                    0.0, 1.0,
                )
                exposure = params.min_size + (params.max_size - params.min_size) * strength
                highest, lowest = -np.inf, np.inf
                floor, ceiling = -np.inf, np.inf
            raw_signal[i] = side
            if side:
                raw_size[i] = exposure

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw_signal
        df['size'] = raw_size
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = GeneratedParams(graph_span=6, memory=8)
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=80, freq='D')
    t = np.arange(len(index), dtype=float)
    close = 100.0 + 0.1 * t + np.sin(t / 2.0)
    data = pd.DataFrame({
        'open': close, 'high': close + 1.0, 'low': close - 1.0,
        'close': close, 'volume': np.full(len(index), 1000000.0),
    }, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    prefix = data.iloc[:60]
    prefix_indicators = strategy.indicators(prefix, params)
    pd.testing.assert_frame_equal(indicators.iloc[:60], prefix_indicators)
    pd.testing.assert_frame_equal(
        signals.iloc[:60],
        strategy.generate_signals(prefix, prefix_indicators, ctx, params).data,
    )
    assert set(signals['signal'].unique()) <= {-1, 0, 1}
    assert np.isfinite(signals['size']).all() and (signals['size'] > 0.0).all()
    # Controlled entry followed by a high-water drawdown verifies the shifted exit.
    entry = strategy.warmup_bars(params)
    controlled = data.copy()
    controlled['close'] = 100.0
    controlled.iloc[entry + 2, controlled.columns.get_loc('close')] = 105.0
    forced = pd.DataFrame({'score': 0.0, 'atr': 2.0}, index=index)
    forced.iloc[entry, forced.columns.get_loc('score')] = 0.6
    result = strategy.generate_signals(controlled, forced, ctx, params).data
    assert result['signal'].iloc[entry:entry + 5].tolist() == [0, 1, 1, 1, 0]
    assert result['size'].iloc[entry + 1] > params.min_size
    forced['score'] = 0.0
    forced.iloc[entry, forced.columns.get_loc('score')] = -0.6
    result = strategy.generate_signals(controlled, forced, ctx, params).data
    assert result['signal'].iloc[entry + 1] == -1
