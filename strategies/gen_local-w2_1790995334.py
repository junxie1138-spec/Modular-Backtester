from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalAnalogParams:
    pattern_window: int = 16
    history_bars: int = 252
    percentile_window: int = 126
    entropy_percentile: float = 0.30
    holding_bars: int = 4
    min_analogs: int = 8
    kernel_width: float = 0.08


class GeneratedStrategy(BaseStrategy[OrdinalAnalogParams]):
    strategy_id = 'gen_local-w2_1790995334'

    @classmethod
    def params_type(cls) -> type[OrdinalAnalogParams]:
        return OrdinalAnalogParams

    def warmup_bars(self, params: OrdinalAnalogParams) -> int:
        return int(params.pattern_window + 3 + max(
            params.history_bars + params.holding_bars,
            params.percentile_window
        ))

    def indicators(self, data: pd.DataFrame, params: OrdinalAnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        log_close = np.log(data['close'].astype(float).where(data['close'] > 0))
        returns = log_close.diff()
        a, b, c = returns.shift(2), returns.shift(1), returns
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        ordinal = (
            4 * (a > b).astype(int)
            + 2 * (a > c).astype(int)
            + (b > c).astype(int)
        ).astype(float).where(valid)
        out['log_close'] = log_close
        out['log_return'] = returns
        out['ordinal'] = ordinal

        frequency_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            name = f'frequency_{code}'
            out[name] = ordinal.eq(code).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
            frequency_columns.append(name)

        frequencies = out[frequency_columns]
        entropy_terms = frequencies * np.log(frequencies.clip(lower=1e-12))
        out['entropy'] = -entropy_terms.sum(axis=1, min_count=6) / np.log(6.0)
        out['entropy_threshold'] = out['entropy'].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entropy_percentile)

        weighted_outcome = pd.Series(0.0, index=data.index)
        total_weight = pd.Series(0.0, index=data.index)
        analog_count = pd.Series(0, index=data.index, dtype=int)
        current_complete = frequencies.notna().all(axis=1)

        # All candidate forward outcomes finish strictly before the decision bar.
        for lag in range(params.holding_bars + 1,
                         params.holding_bars + params.history_bars + 1):
            historical = frequencies.shift(lag)
            distance = (frequencies - historical).pow(2).sum(axis=1, min_count=6)
            outcome = log_close.shift(lag - params.holding_bars) - log_close.shift(lag)
            eligible = (
                current_complete
                & historical.notna().all(axis=1)
                & ordinal.eq(ordinal.shift(lag))
                & np.isfinite(distance)
                & np.isfinite(outcome)
            )
            weight = np.exp(-distance / max(params.kernel_width, 1e-12)).where(
                eligible, 0.0
            ).fillna(0.0)
            weighted_outcome += weight * outcome.fillna(0.0)
            total_weight += weight
            analog_count += eligible.astype(int)

        out['analog_count'] = analog_count
        out['analog_weight'] = total_weight
        out['forecast'] = (weighted_outcome / total_weight.where(total_weight > 0)).where(
            analog_count >= params.min_analogs
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalAnalogParams) -> SignalFrame:
        eligible = (
            indicators['entropy'].le(indicators['entropy_threshold'])
            & np.isfinite(indicators['forecast'])
            & indicators['entropy_threshold'].notna()
        )
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[eligible & indicators['forecast'].gt(0)] = 1
        candidates.loc[eligible & indicators['forecast'].lt(0)] = -1
        candidates.iloc[:self.warmup_bars(params)] = 0

        signal = np.zeros(len(data), dtype=int)
        active = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates.to_numpy(dtype=int)):
            if active != 0:
                if i >= exit_bar:
                    active = 0
            elif candidate != 0:
                active = int(candidate)
                exit_bar = i + params.holding_bars
            signal[i] = active

        df = pd.DataFrame(index=data.index)
        df['signal'] = signal
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
