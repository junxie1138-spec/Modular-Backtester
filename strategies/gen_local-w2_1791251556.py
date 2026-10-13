from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ordinal_window: int = 24
    complexity_lag: int = 5
    entropy_jump: float = 0.12
    prior_entropy_max: float = 0.80
    motif_bias_min: float = 0.20
    rejection_window: int = 3
    rejection_min: float = 0.12
    atr_window: int = 14
    stop_atr: float = 2.0
    vol_window: int = 20
    target_vol: float = 0.12
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791251556'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.ordinal_window + params.complexity_lag + 3,
            params.atr_window + 1,
            params.vol_window + 1,
            params.rejection_window,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna()
        valid &= a.ne(b) & a.ne(c) & b.ne(c)
        code = (4 * a.gt(b).astype(int)
                + 2 * a.gt(c).astype(int)
                + b.gt(c).astype(int))
        out['ordinal_code'] = code.where(valid)
        probability_columns = []
        for motif in (0, 1, 3, 4, 6, 7):
            name = f'probability_{motif}'
            observation = code.eq(motif).astype(float).where(valid)
            out[name] = observation.rolling(params.ordinal_window).mean()
            probability_columns.append(name)
        probabilities = out[probability_columns]
        terms = probabilities * np.log(probabilities.where(probabilities.gt(0)))
        out['entropy'] = -terms.fillna(0).sum(axis=1) / np.log(6.0)
        out['entropy'] = out['entropy'].where(probabilities.notna().all(axis=1))
        out['prior_entropy'] = out['entropy'].shift(params.complexity_lag)
        out['entropy_change'] = out['entropy'] - out['prior_entropy']
        out['motif_bias'] = out['probability_0'] - out['probability_7']
        out['prior_bias'] = out['motif_bias'].shift(params.complexity_lag)

        out['lower_wick'] = (data[['open', 'close']].min(axis=1) - data['low']).clip(lower=0)
        out['upper_wick'] = (data['high'] - data[['open', 'close']].max(axis=1)).clip(lower=0)
        out['range'] = (data['high'] - data['low']).clip(lower=0)
        volume = data['volume'].clip(lower=0)
        out['weighted_rejection'] = (out['lower_wick'] - out['upper_wick']) * volume
        out['weighted_range'] = out['range'] * volume
        numerator = out['weighted_rejection'].rolling(params.rejection_window).sum()
        denominator = out['weighted_range'].rolling(params.rejection_window).sum()
        out['rejection'] = numerator / denominator.replace(0, np.nan)

        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            out['range'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['return'] = close.pct_change(fill_method=None)
        out['annual_vol'] = out['return'].rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out['target_size'] = (params.target_vol / out['annual_vol'].replace(0, np.nan)).clip(0.05, 0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        complexity_transition = (
            indicators['prior_entropy'].le(params.prior_entropy_max)
            & indicators['entropy_change'].ge(params.entropy_jump)
        )
        long_setup = (
            complexity_transition
            & indicators['prior_bias'].le(-params.motif_bias_min)
            & indicators['rejection'].ge(params.rejection_min)
        )
        short_setup = (
            complexity_transition
            & indicators['prior_bias'].ge(params.motif_bias_min)
            & indicators['rejection'].le(-params.rejection_min)
        )
        candidates = np.select([long_setup, short_setup], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        volatility = indicators['annual_vol'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_price = np.nan
        entry_atr = np.nan
        held = 0
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if side:
                # The previous close's decision fills at this bar's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    entry_price = opens[i]
                held += 1
                fixed_stop_hit = (
                    np.isfinite(closes[i])
                    and side * (closes[i] - entry_price) <= -params.stop_atr * entry_atr
                )
                if fixed_stop_hit or held >= params.hold_bars:
                    side = 0
                    entry_price = np.nan
                    held = 0
                    continue
                signals[i] = side
            elif (
                i >= warmup - 1
                and candidates[i] != 0
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(volatility[i]) and volatility[i] > 0
            ):
                side = int(candidates[i])
                entry_price = np.nan
                entry_atr = atr[i]
                held = 0
                signals[i] = side

        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = indicators['target_size'].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
