from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    momentum_period: int = 20
    rank_period: int = 63
    momentum_threshold: float = 0.65
    efficiency_threshold: float = 0.62
    atr_period: int = 14
    breakeven_pct: float = 1.5
    trail_k: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_jay_1778893684'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.rank_period + params.momentum_period + params.atr_period + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data['close']
        high = data['high']
        low = data['low']

        # Primitive 1: N-bar momentum (raw return)
        momentum = close.pct_change(params.momentum_period)

        # Primitive 2: signed Kaufman Efficiency Ratio
        # ER = net displacement / path length (0=choppy, 1=straight-line)
        net_displacement = (close - close.shift(params.momentum_period)).abs()
        path_length = close.diff().abs().rolling(params.momentum_period).sum()
        efficiency = net_displacement / path_length.replace(0.0, np.nan)
        # Sign by momentum direction: +1 = efficient uptrend, -1 = efficient downtrend
        directional_efficiency = efficiency * np.sign(momentum)

        # Rolling percentile rank of each primitive over rank_period bars
        mom_rank = momentum.rolling(params.rank_period).rank(pct=True)
        eff_rank = directional_efficiency.rolling(params.rank_period).rank(pct=True)

        # ATR for trailing stop
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period).mean()

        return pd.DataFrame(
            {'mom_rank': mom_rank, 'eff_rank': eff_rank, 'atr': atr},
            index=data.index,
        )

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = data.copy()

        mom_rank = indicators['mom_rank'].values
        eff_rank = indicators['eff_rank'].values
        atr_vals = indicators['atr'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)

        upper_m = params.momentum_threshold
        lower_m = 1.0 - params.momentum_threshold
        upper_e = params.efficiency_threshold
        lower_e = 1.0 - params.efficiency_threshold
        bp_mult = 1.0 + params.breakeven_pct / 100.0
        bm_mult = 1.0 - params.breakeven_pct / 100.0

        sig_out = np.zeros(n, dtype=np.int64)
        position = 0
        entry_price = 0.0
        stop_price = 0.0
        breakeven_hit = False

        for i in range(n):
            mr = mom_rank[i]
            er = eff_rank[i]
            at = atr_vals[i]
            c = closes[i]
            h = highs[i]
            lo = lows[i]

            if np.isnan(mr) or np.isnan(er) or np.isnan(at):
                sig_out[i] = 0
                continue

            if position == 0:
                # Both primitives must agree for entry
                if mr > upper_m and er > upper_e:
                    position = 1
                    entry_price = c
                    stop_price = c - params.trail_k * at
                    breakeven_hit = False
                    sig_out[i] = 1
                elif mr < lower_m and er < lower_e:
                    position = -1
                    entry_price = c
                    stop_price = c + params.trail_k * at
                    breakeven_hit = False
                    sig_out[i] = -1
                # else: remain flat

            elif position == 1:
                # Promote stop to breakeven once +breakeven_pct% reached via high
                if not breakeven_hit and h >= entry_price * bp_mult:
                    breakeven_hit = True
                    if entry_price > stop_price:
                        stop_price = entry_price
                # Trail stop upward only
                trail = h - params.trail_k * at
                if trail > stop_price:
                    stop_price = trail
                # Stop check on close
                if c <= stop_price:
                    position = 0
                    sig_out[i] = 0
                else:
                    sig_out[i] = 1

            else:  # position == -1
                # Promote stop to breakeven once -breakeven_pct% reached via low
                if not breakeven_hit and lo <= entry_price * bm_mult:
                    breakeven_hit = True
                    if entry_price < stop_price:
                        stop_price = entry_price
                # Trail stop downward only
                trail = lo + params.trail_k * at
                if trail < stop_price:
                    stop_price = trail
                # Stop check on close
                if c >= stop_price:
                    position = 0
                    sig_out[i] = 0
                else:
                    sig_out[i] = -1

        signal = pd.Series(sig_out, index=df.index, dtype=int)
        # Mandatory 1-bar shift: decision at bar N executes at bar N+1
        signal = signal.shift(1).fillna(0).astype(int)

        df['signal'] = signal
        df['size'] = 1.0

        return SignalFrame(data=df, signal_column='signal', size_column='size')
