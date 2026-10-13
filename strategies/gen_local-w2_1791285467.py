from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    pattern_window: int = 30
    mean_window: int = 40
    atr_window: int = 14
    displacement_atr: float = 1.2
    entropy_rebound: float = 0.06
    entropy_ceiling: float = 0.85
    descending_share: float = 0.32
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = 'gen_local-w2_1791285467'

    @classmethod
    def params_type(cls) -> type[OrdinalParams]:
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return int(max(params.mean_window, params.pattern_window + 5,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out['equilibrium'] = close.rolling(
            params.mean_window, min_periods=params.mean_window).mean()
        out['displacement'] = (close - out['equilibrium']) / out['atr'].replace(0, np.nan)

        a, b, c = close.shift(2), close.shift(1), close
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        # Chronological tie-breaking gives each triple one of six orderings.
        code = ((a <= b).astype(int) + 2 * (a <= c).astype(int)
                + 4 * (b <= c).astype(int))
        out['ordinal_code'] = code.where(valid)
        probability_columns = []
        for value in (0, 1, 3, 4, 6, 7):
            name = 'pattern_probability_' + str(value)
            out[name] = (out['ordinal_code'] == value).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window).mean()
            probability_columns.append(name)
        probabilities = out[probability_columns]
        out['entropy'] = -(probabilities * np.log(
            probabilities.clip(lower=1e-12))).sum(axis=1, min_count=6) / np.log(6.0)
        out['prior_entropy'] = out['entropy'].shift(3)
        out['entropy_change'] = out['entropy'] - out['prior_entropy']
        out['descending_probability'] = out['pattern_probability_0']
        out['candidate'] = (
            (out['displacement'] <= -params.displacement_atr)
            & (out['prior_entropy'] <= params.entropy_ceiling)
            & (out['entropy_change'] >= params.entropy_rebound)
            & (out['descending_probability'] >= params.descending_share)
            & (out['atr'] > 0)
        ).fillna(False)
        out['confirmed'] = out['candidate'] & out['candidate'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        frozen_atr = 0.0
        fixed_stop = 0.0
        held = 0

        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                # The prior close's decision fills at this bar's open.
                fixed_stop = opens[i] - params.stop_atr * frozen_atr
                active = True
                held = 0

            if active:
                held += 1
                if (not np.isfinite(closes[i])
                        or closes[i] < fixed_stop
                        or held >= params.hold_bars):
                    active = False
                    continue
                decisions[i] = 1
            elif (confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0
                  and np.isfinite(closes[i]) and closes[i] > 0):
                frozen_atr = atr[i]
                pending = True
                decisions[i] = 1

        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')
