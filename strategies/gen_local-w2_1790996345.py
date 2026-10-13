from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapOrdinalParams:
    ordinal_window: int = 24
    atr_window: int = 14
    pressure_window: int = 5
    complexity_floor: float = 0.06
    growth_margin: float = 0.08
    gap_atr: float = 0.35
    rejection_fraction: float = 0.60
    trail_atr: float = 2.5
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GapOrdinalParams]):
    strategy_id = 'gen_local-w2_1790996345'

    @classmethod
    def params_type(cls) -> type[GapOrdinalParams]:
        return GapOrdinalParams

    def warmup_bars(self, params: GapOrdinalParams) -> int:
        return int(max(params.ordinal_window + 3,
                       params.atr_window + 1,
                       params.pressure_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapOrdinalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data['close'].shift(1)
        out['gap_return'] = (data['open'] / previous.where(previous > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out['gap_points'] = data['open'] - previous
        out['body_points'] = data['close'] - data['open']
        out['absolute_gap'] = out['gap_return'].abs()
        a = out['absolute_gap'].shift(2)
        b = out['absolute_gap'].shift(1)
        c = out['absolute_gap']
        valid = a.notna() & b.notna() & c.notna()
        # Equal magnitudes are ordered by their observation time.
        conditions = [
            (a <= b) & (b <= c),
            (a <= c) & (c < b),
            (b < a) & (a <= c),
            (b <= c) & (c < a),
            (c < a) & (a <= b),
            (c < b) & (b < a),
        ]
        out['ordinal_pattern'] = pd.Series(
            np.select(conditions, [0, 1, 2, 3, 4, 5], default=-1),
            index=data.index, dtype=float
        ).where(valid)
        for code in range(6):
            observations = out['ordinal_pattern'].eq(code).astype(float).where(valid)
            out['prob_' + str(code)] = observations.rolling(
                params.ordinal_window, min_periods=params.ordinal_window
            ).mean()
        probabilities = out[['prob_' + str(code) for code in range(6)]]
        complete = probabilities.notna().all(axis=1)
        log_probabilities = np.log(probabilities.clip(lower=1e-12))
        out['entropy'] = (-(probabilities * log_probabilities).sum(axis=1) / np.log(6.0)).where(complete)
        out['disequilibrium'] = (((probabilities - 1.0 / 6.0) ** 2).sum(axis=1) * 6.0 / 5.0).where(complete)
        out['ordinal_complexity'] = out['entropy'] * out['disequilibrium']
        out['amplification_balance'] = out['prob_0'] - out['prob_5']
        out['gap_pressure'] = out['gap_return'].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        ranges = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1)
        out['true_range'] = ranges.max(axis=1).where(previous.notna())
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out['gap_strength'] = out['gap_points'].abs() / out['atr'].where(out['atr'] > 0)
        out['rejection_ratio'] = out['body_points'].abs() / out['gap_points'].abs().where(out['gap_points'].abs() > 0)
        organized = (
            complete
            & out['ordinal_complexity'].ge(params.complexity_floor)
            & out['amplification_balance'].ge(params.growth_margin)
        )
        out['ordinal_vote'] = np.select([
            organized & out['gap_pressure'].gt(0),
            organized & out['gap_pressure'].lt(0),
        ], [-1, 1], default=0).astype(int)
        substantial = (
            out['gap_strength'].ge(params.gap_atr)
            & out['rejection_ratio'].ge(params.rejection_fraction)
        )
        out['rejection_vote'] = np.select([
            substantial & out['gap_points'].gt(0) & out['body_points'].lt(0),
            substantial & out['gap_points'].lt(0) & out['body_points'].gt(0),
        ], [-1, 1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapOrdinalParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        ready = indicators[['atr', 'ordinal_complexity', 'gap_pressure',
                            'gap_strength', 'rejection_ratio']].notna().all(axis=1)
        agrees = indicators['ordinal_vote'].eq(indicators['rejection_vote'])
        entries = indicators['ordinal_vote'].where(ready & agrees, 0).fillna(0).astype(int).to_numpy()
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        trailing_floor = np.nan
        short_ceiling = np.nan
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                high_water = np.nan
                trailing_floor = np.nan
                short_ceiling = np.nan
                continue
            if position != 0:
                age += 1
                high_water = max(high_water, close)
                # The required highest-close floor applies to either direction.
                # Max preserves its ratchet even when ATR increases.
                trailing_floor = max(trailing_floor, high_water - params.trail_atr * atr)
                exit_now = close <= trailing_floor or age >= params.max_hold
                # A short also needs an adverse-price exit; freeze it at entry.
                if position == -1 and close >= short_ceiling:
                    exit_now = True
                if exit_now:
                    position = 0
                    age = 0
                    high_water = np.nan
                    trailing_floor = np.nan
                    short_ceiling = np.nan
                    signals[i] = 0
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                age = 0
                high_water = close
                trailing_floor = close - params.trail_atr * atr
                short_ceiling = close + params.trail_atr * atr
            signals[i] = position
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')
