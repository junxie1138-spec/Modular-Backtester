from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GraphParams:
    graph_window: int = 24
    edge_threshold: float = 0.18
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    holding_bars: int = 10


class GeneratedStrategy(BaseStrategy[GraphParams]):
    strategy_id = 'gen_local_1791263923'

    @classmethod
    def params_type(cls) -> type[GraphParams]:
        return GraphParams

    def warmup_bars(self, params: GraphParams) -> int:
        return max(params.graph_window, params.atr_window) + 1

    def indicators(self, data: pd.DataFrame, params: GraphParams) -> pd.DataFrame:
        if params.graph_window < 3 or params.atr_window < 2:
            raise ValueError('Indicator windows must be at least three and two bars.')
        if not 0.0 < params.edge_threshold < 1.0:
            raise ValueError('edge_threshold must lie between zero and one.')
        if params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError('Breakeven and trailing parameters must be positive.')
        if not 5 <= params.holding_bars <= 10:
            raise ValueError('holding_bars must be between five and ten.')

        close = data['close'].where(data['close'] > 0)
        returns = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        magnitude = returns.abs()
        signs = np.sign(returns)
        numerator = pd.Series(0.0, index=data.index)
        denominator = pd.Series(0.0, index=data.index)
        window = params.graph_window

        # Each rolling sum includes precisely the edges inside the current graph.
        for lag in range(1, window):
            if lag == 1:
                visible = returns.notna() & returns.shift(lag).notna()
            else:
                interior = magnitude.shift(1).rolling(lag - 1).max()
                visible = (magnitude > interior) & (magnitude.shift(lag) > interior)
            weight = visible.astype(float) * float(lag)
            agreement = (signs + signs.shift(lag)) * 0.5
            numerator += (weight * agreement.fillna(0.0)).rolling(window - lag).sum().fillna(0.0)
            denominator += weight.rolling(window - lag).sum().fillna(0.0)

        ready = returns.rolling(window).count().eq(window)
        score = (numerator / denominator.replace(0.0, np.nan)).where(ready, 0.0).fillna(0.0)
        momentum = returns.rolling(window).sum().fillna(0.0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean().fillna(0.0)
        return pd.DataFrame({
            'returns': returns.fillna(0.0),
            'magnitude': magnitude.fillna(0.0),
            'return_sign': signs.fillna(0.0),
            'edge_numerator': numerator,
            'edge_denominator': denominator,
            'graph_score': score.clip(-1.0, 1.0),
            'momentum': momentum,
            'true_range': true_range.fillna(0.0),
            'atr': atr,
            'ready': ready,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GraphParams) -> SignalFrame:
        score = indicators['graph_score'].to_numpy(dtype=float)
        momentum = indicators['momentum'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ready = indicators['ready'].to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        direction = 0
        pending = False
        armed = False
        entry = peak = stop = entry_atr = 0.0
        exposure = 0.25
        age = 0

        for i in range(len(data)):
            exited = False
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry = opens[i]
                    peak = direction * entry
                    stop = peak - params.trail_atr * entry_atr
                    age = 0
                    armed = False
                else:
                    direction = 0
                    exited = True

            if direction:
                age += 1
                if np.isfinite(closes[i]) and closes[i] > 0:
                    favorable_price = direction * closes[i]
                    # Signed coordinates make the stop ratchet upward for either side.
                    if favorable_price <= stop or age >= params.holding_bars:
                        direction = 0
                        exited = True
                    else:
                        peak = max(peak, favorable_price)
                        if direction * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                            armed = True
                        if armed:
                            current_atr = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else entry_atr
                            stop = max(stop, direction * entry, peak - params.trail_atr * current_atr)
                elif age >= params.holding_bars:
                    direction = 0
                    exited = True

            if not direction and not exited and ready[i] and atr[i] > 0:
                candidate = 1 if score[i] >= params.edge_threshold else -1 if score[i] <= -params.edge_threshold else 0
                if candidate and candidate * momentum[i] > 0:
                    direction = candidate
                    pending = True
                    entry_atr = atr[i]
                    strength = (abs(score[i]) - params.edge_threshold) / (1.0 - params.edge_threshold)
                    exposure = float(np.clip(0.25 + 0.70 * strength, 0.25, 0.95))

            signals[i] = direction
            sizes[i] = exposure if direction else 0.25

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = GraphParams()
    for side in (1, -1):
        close = 100.0 * np.exp(side * np.arange(100) * 0.003)
        data = pd.DataFrame({'open': close, 'high': close * 1.002,
                             'low': close * 0.998, 'close': close, 'volume': 1000.0},
                            index=pd.date_range('2020-01-01', periods=100))
        indicators = strategy.indicators(data, params)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].isin([-1, 0, 1]).all() and result['size'].gt(0).all()
        assert result['signal'].iloc[:params.graph_window + 1].eq(0).all()
        assert result['signal'].eq(side).any() and result['signal'].iloc[35] == 0
        prefix = strategy.generate_signals(data.iloc[:60], strategy.indicators(data.iloc[:60], params), None, params).data
        pd.testing.assert_frame_equal(result.iloc[:60], prefix)
