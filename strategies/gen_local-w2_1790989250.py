from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalBoundaryParams:
    pattern_window: int = 20
    complexity_gap: float = 0.12
    ascending_floor_share: float = 0.40
    minimum_ceiling_complexity: float = 0.45
    volatility_window: int = 20
    target_volatility: float = 0.12
    atr_window: int = 14
    initial_stop_atr: float = 2.0
    breakeven_gain: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 10
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[OrdinalBoundaryParams]):
    strategy_id = 'gen_local-w2_1790989250'

    @classmethod
    def params_type(cls) -> type[OrdinalBoundaryParams]:
        return OrdinalBoundaryParams

    def warmup_bars(self, params: OrdinalBoundaryParams) -> int:
        return int(max(200, max(3, params.pattern_window) + 2,
                       max(2, params.volatility_window) + 1,
                       max(2, params.atr_window) + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalBoundaryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['returns'] = returns
        vol_window = max(2, int(params.volatility_window))
        out['annual_volatility'] = returns.rolling(vol_window, min_periods=vol_window).std(ddof=0) * np.sqrt(252.0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        out['true_range'] = true_range
        atr_window = max(2, int(params.atr_window))
        out['atr'] = true_range.rolling(atr_window, min_periods=atr_window).mean()
        window = max(3, int(params.pattern_window))
        states = (1, 2, 3, 5, 6, 7)
        for boundary, column in (('ceiling', 'high'), ('floor', 'low')):
            x = data[column].astype(float)
            a, b, c = x.shift(2), x.shift(1), x
            valid = a.notna() & b.notna() & c.notna()
            # Stable chronological tie breaking yields six permutation states.
            rank_a = (b < a).astype(int) + (c < a).astype(int)
            rank_b = (a <= b).astype(int) + (c < b).astype(int)
            ordinal = (3 * rank_a + rank_b).where(valid)
            out[boundary + '_ordinal'] = ordinal
            probability_columns = []
            for state in states:
                name = boundary + '_p' + str(state)
                membership = ordinal.eq(state).astype(float).where(valid)
                out[name] = membership.rolling(window, min_periods=window).mean()
                probability_columns.append(name)
            probabilities = out[probability_columns]
            # Normalized Gini-Simpson diversity: zero ordered, one uniform.
            diversity = (1.0 - probabilities.pow(2).sum(axis=1, min_count=6)) * 1.2
            out[boundary + '_complexity'] = diversity.clip(0.0, 1.0)
        out['complexity_gap'] = out['ceiling_complexity'] - out['floor_complexity']
        out['ascending_floor_share'] = out['floor_p1']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalBoundaryParams) -> SignalFrame:
        required = ['ma200', 'annual_volatility', 'atr', 'ceiling_complexity',
                    'floor_complexity', 'complexity_gap', 'ascending_floor_share']
        ready = indicators[required].notna().all(axis=1)
        ready &= indicators['atr'].gt(0.0) & indicators['annual_volatility'].gt(0.0)
        regime = data['close'].gt(indicators['ma200'])
        entry_condition = (
            ready & regime
            & indicators['complexity_gap'].ge(params.complexity_gap)
            & indicators['ascending_floor_share'].ge(params.ascending_floor_share)
            & indicators['ceiling_complexity'].ge(params.minimum_ceiling_complexity)
        ).fillna(False).to_numpy(dtype=bool)
        exposure = (max(0.001, float(params.target_volatility))
                    / indicators['annual_volatility'].clip(lower=0.001))
        exposure = exposure.clip(lower=0.01, upper=max(0.01, min(0.95, float(params.max_size))))
        exposure = exposure.replace([np.inf, -np.inf], np.nan).fillna(0.01)
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ma = indicators['ma200'].to_numpy(dtype=float)
        valid = ready.to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        entry_price = np.nan
        stop = np.nan
        peak = np.nan
        armed = False
        entry_bar = -1
        max_hold = max(1, int(params.max_hold_bars))
        initial_multiple = max(0.1, float(params.initial_stop_atr))
        trail_multiple = max(0.1, float(params.trail_atr))
        activation = max(0.001, float(params.breakeven_gain))
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                # The preceding close's decision fills at this bar's open.
                if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0.0 or not np.isfinite(atr[i - 1]):
                    active = False
                    continue
                entry_price = opens[i]
                entry_bar = i
                peak = entry_price
                stop = entry_price - initial_multiple * atr[i - 1]
                armed = False
                active = True
            if active:
                if not valid[i] or not np.isfinite(closes[i]) or not np.isfinite(highs[i]) or not np.isfinite(lows[i]):
                    active = False
                    continue
                # Existing stop breaches produce an exit decision at this close.
                # Execution remains on the following bar, including gap risk.
                exit_now = lows[i] <= stop
                exit_now = exit_now or closes[i] <= ma[i]
                exit_now = exit_now or (i - entry_bar + 1 >= max_hold)
                if exit_now:
                    active = False
                    continue
                peak = max(peak, highs[i])
                if highs[i] >= entry_price * (1.0 + activation):
                    armed = True
                if armed:
                    # Neither changing ATR nor lower highs can loosen this stop.
                    stop = max(stop, entry_price, peak - trail_multiple * atr[i])
                    if closes[i] <= stop:
                        active = False
                        continue
                signal[i] = 1
            elif entry_condition[i]:
                signal[i] = 1
                pending_entry = True
        df = pd.DataFrame(index=data.index)
        df['signal'] = signal
        df['size'] = exposure.astype(float)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
