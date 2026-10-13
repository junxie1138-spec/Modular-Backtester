from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRegimeParams:
    pattern_window: int = 24
    entropy_split: float = 0.70
    complexity_floor: float = 0.035
    ordinal_threshold: float = 0.08
    pressure_window: int = 3
    pressure_threshold: float = 0.12
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[OrdinalRegimeParams]):
    strategy_id = 'gen_local-w2_1790988468'

    @classmethod
    def params_type(cls) -> type[OrdinalRegimeParams]:
        return OrdinalRegimeParams

    def warmup_bars(self, params: OrdinalRegimeParams) -> int:
        return int(max(params.pattern_window + 3, params.atr_window + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: OrdinalRegimeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['return'] = close.div(close.shift(1).replace(0.0, np.nan)).sub(1.0)
        out['return_oldest'] = out['return'].shift(2)
        out['return_middle'] = out['return'].shift(1)
        a = out['return_oldest']
        b = out['return_middle']
        c = out['return']
        conditions = [
            (a < b) & (b < c),
            (a < c) & (c < b),
            (b < a) & (a < c),
            (b < c) & (c < a),
            (c < a) & (a < b),
            (c < b) & (b < a),
        ]
        out['ordinal_code'] = np.select(conditions, [0, 1, 2, 3, 4, 5], default=np.nan)
        out['valid_patterns'] = out['ordinal_code'].notna().astype(float).rolling(
            params.pattern_window, min_periods=params.pattern_window
        ).sum()
        complete = out['valid_patterns'].eq(params.pattern_window)
        probability_columns = []
        for code in range(6):
            column = f'probability_{code}'
            out[column] = out['ordinal_code'].eq(code).astype(float).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean().where(complete)
            probability_columns.append(column)
        probabilities = out[probability_columns]
        log_probabilities = np.log(probabilities.clip(lower=1e-12))
        out['entropy'] = -(probabilities * log_probabilities).sum(
            axis=1, min_count=6
        ) / np.log(6.0)
        out['disequilibrium'] = probabilities.sub(1.0 / 6.0).pow(2).sum(
            axis=1, min_count=6
        ) / (5.0 / 6.0)
        out['complexity'] = out['entropy'] * out['disequilibrium']
        out['ordinal_bias'] = out['probability_0'] - out['probability_5']
        out['range'] = (data['high'] - data['low']).astype(float)
        out['high_previous_close'] = (data['high'] - close.shift(1)).abs()
        out['low_previous_close'] = (data['low'] - close.shift(1)).abs()
        out['true_range'] = out[
            ['range', 'high_previous_close', 'low_previous_close']
        ].max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out['close_location'] = (
            (2.0 * close - data['high'] - data['low'])
            / out['range'].replace(0.0, np.nan)
        ).clip(-1.0, 1.0).where(out['range'].ne(0.0), 0.0)
        out['valid_volume'] = data['volume'].astype(float).clip(lower=0.0)
        out['signed_volume'] = out['close_location'] * out['valid_volume']
        out['pressure_numerator'] = out['signed_volume'].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out['pressure_denominator'] = out['valid_volume'].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out['pressure'] = out['pressure_numerator'].div(
            out['pressure_denominator'].replace(0.0, np.nan)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalRegimeParams,
    ) -> SignalFrame:
        entropy = indicators['entropy']
        bias = indicators['ordinal_bias']
        ordered = entropy.le(params.entropy_split)
        complex_regime = entropy.gt(params.entropy_split) & indicators['complexity'].ge(
            params.complexity_floor
        )
        ordinal_long = (ordered & bias.ge(params.ordinal_threshold)) | (
            complex_regime & bias.le(-params.ordinal_threshold)
        )
        ordinal_short = (ordered & bias.le(-params.ordinal_threshold)) | (
            complex_regime & bias.ge(params.ordinal_threshold)
        )
        pressure = indicators['pressure']
        valid = indicators[
            ['entropy', 'complexity', 'ordinal_bias', 'pressure', 'atr']
        ].notna().all(axis=1) & indicators['atr'].gt(0.0)
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[valid & ordinal_long & pressure.ge(params.pressure_threshold)] = 1
        candidates.loc[valid & ordinal_short & pressure.le(-params.pressure_threshold)] = -1
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = candidates.to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        awaiting_fill = False
        age = 0
        for i in range(len(data)):
            if position != 0:
                if awaiting_fill:
                    entry_price = opens[i]
                    awaiting_fill = False
                age += 1
                usable = (
                    np.isfinite(entry_price) and entry_price > 0.0
                    and np.isfinite(closes[i]) and np.isfinite(entry_atr)
                    and entry_atr > 0.0
                )
                stopped = False
                if usable:
                    if position == 1:
                        stopped = closes[i] < entry_price - params.stop_atr * entry_atr
                    else:
                        stopped = closes[i] > entry_price + params.stop_atr * entry_atr
                if not usable or stopped or age >= params.holding_bars:
                    position = 0
                    entry_price = np.nan
                    entry_atr = np.nan
                    raw[i] = 0
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                entry_atr = atrs[i]
                awaiting_fill = True
                age = 0
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')
