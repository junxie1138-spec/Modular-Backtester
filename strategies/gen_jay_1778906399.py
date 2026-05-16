from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    infection_window: int = 10
    vol_baseline: int = 20
    return_window: int = 3
    vol_surge_mult: float = 1.3
    profit_target_pct: float = 1.5
    time_stop_bars: int = 3


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_jay_1778906399'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(200, params.vol_baseline, params.infection_window, params.return_window) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data['close']
        volume = data['volume']

        ma_200 = close.rolling(200, min_periods=200).mean()
        bull_regime = (close > ma_200).fillna(False)

        # SI model: classify each bar's volume into infected (up) vs susceptible (down)
        up_bar = (close > close.shift(1)).fillna(False).astype(float)
        up_vol = volume * up_bar
        down_vol = volume * (1.0 - up_bar)

        # Rolling infected and susceptible population totals
        inf_pop = up_vol.rolling(params.infection_window, min_periods=1).sum()
        sus_pop = down_vol.rolling(params.infection_window, min_periods=1).sum()

        # SI transmission term: I*S/(I+S)^2 — peaks when I==S (maximum epidemic growth rate)
        total_pop = (inf_pop + sus_pop).replace(0, np.nan)
        si_term = (inf_pop * sus_pop) / (total_pop ** 2)

        # Positive slope = epidemic still in growth phase, not yet saturated
        si_slope = si_term.diff()

        # Infected fraction: what share of recent volume is bullish
        inf_frac = inf_pop / total_pop

        # Volume surge: current bar vs rolling baseline
        vol_mean = volume.rolling(params.vol_baseline, min_periods=1).mean().replace(0, np.nan)
        vol_ratio = volume / vol_mean

        # Multi-bar price momentum
        price_return = close.pct_change(params.return_window)

        # Entry conditions (unshifted — shift applied in generate_signals via raw_entry[i-1]):
        # 1. SI transmission term still rising (epidemic growth phase)
        # 2. Infected fraction in [0.3, 0.75]: momentum building but not yet saturated
        # 3. Volume surge confirms new participant inflow
        # 4. Positive price return validates directional move
        # 5. Above 200-day MA (bull regime)
        raw_entry = (
            (si_slope > 0)
            & (inf_frac > 0.3)
            & (inf_frac < 0.75)
            & (vol_ratio > params.vol_surge_mult)
            & (price_return > 0.0)
            & bull_regime
        ).astype(int)

        return pd.DataFrame(
            {
                'ma_200': ma_200,
                'si_term': si_term,
                'si_slope': si_slope,
                'inf_frac': inf_frac,
                'vol_ratio': vol_ratio,
                'price_return': price_return,
                'raw_entry': raw_entry,
            },
            index=data.index,
        )

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data['close'].values
        raw_entry = indicators['raw_entry'].fillna(0).astype(int).values
        n = len(close)

        signal = np.zeros(n, dtype=int)
        size_arr = np.ones(n, dtype=float)

        in_position = False
        entry_price = 0.0
        bars_held = 0
        profit_target = params.profit_target_pct / 100.0

        for i in range(n):
            if in_position:
                bars_held += 1
                gain = (close[i] - entry_price) / entry_price if entry_price > 0.0 else 0.0
                if gain >= profit_target or bars_held >= params.time_stop_bars:
                    # Profit target or time stop fired — exit (signal stays 0)
                    in_position = False
                    bars_held = 0
                else:
                    signal[i] = 1

            # 1-bar shift: act on previous bar's raw_entry to avoid lookahead
            if not in_position and i > 0 and raw_entry[i - 1] == 1:
                signal[i] = 1
                in_position = True
                entry_price = close[i]
                bars_held = 0

        df = pd.DataFrame({'signal': signal, 'size': size_arr}, index=data.index)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
