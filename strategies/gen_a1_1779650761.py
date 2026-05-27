from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTideParams:
    range_mean_window: int = 20
    atr_window: int = 14
    range_contraction_ratio: float = 0.9
    close_position_threshold: float = 0.66
    min_gap_pct: float = 0.001
    breakeven_trigger_pct: float = 0.015
    atr_trail_mult: float = 2.0
    max_hold_bars: int = 5
    trend_filter_window: int = 200


class GeneratedStrategy(BaseStrategy[GapTideParams]):
    strategy_id = 'gen_a1_1779650761'

    @classmethod
    def params_type(cls):
        return GapTideParams

    @classmethod
    def warmup_bars(cls, params: GapTideParams) -> int:
        return max(params.range_mean_window, params.atr_window, params.trend_filter_window) + 2

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GapTideParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        open_ = data['open']
        high = data['high']
        low = data['low']
        close = data['close']

        prev_close = close.shift(1)
        gap_pct = (open_ - prev_close) / prev_close.replace(0, np.nan)
        ind['gap_pct'] = gap_pct

        tr_hl = high - low
        tr_hc = (high - prev_close).abs()
        tr_lc = (low - prev_close).abs()
        tr = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        ind['atr'] = atr

        rng = high - low
        mean_rng = rng.rolling(params.range_mean_window, min_periods=params.range_mean_window).mean()
        ind['range_ratio'] = rng / mean_rng.replace(0, np.nan)

        denom = (high - low).replace(0, np.nan)
        close_pos = (close - low) / denom
        ind['close_pos'] = close_pos

        coherent_up = (
            (gap_pct > params.min_gap_pct)
            & ((close - open_) > 0)
            & (close_pos > params.close_position_threshold)
            & (ind['range_ratio'] <= params.range_contraction_ratio)
        )
        coherent_down = (
            (gap_pct < -params.min_gap_pct)
            & ((close - open_) < 0)
            & (close_pos < (1.0 - params.close_position_threshold))
            & (ind['range_ratio'] <= params.range_contraction_ratio)
        )
        ind['coherent_up'] = coherent_up.fillna(False).astype(bool)
        ind['coherent_down'] = coherent_down.fillna(False).astype(bool)

        ind['trend_sma'] = close.rolling(params.trend_filter_window, min_periods=params.trend_filter_window).mean()

        return ind

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapTideParams) -> SignalFrame:
        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        atr = indicators['atr'].values.astype(float)
        coherent_up = indicators['coherent_up'].values.astype(bool)
        coherent_down = indicators['coherent_down'].values.astype(bool)
        trend_sma = indicators['trend_sma'].values.astype(float)

        n = len(close)
        raw_signal = np.zeros(n, dtype=int)

        confirmed_long = np.zeros(n, dtype=bool)
        confirmed_short = np.zeros(n, dtype=bool)
        if n > 1:
            confirmed_long[1:] = coherent_up[1:] & coherent_up[:-1]
            confirmed_short[1:] = coherent_down[1:] & coherent_down[:-1]

        position = 0
        entry_price = 0.0
        entry_bar = -1
        stop_price = 0.0
        breakeven_armed = False

        for i in range(n):
            if np.isnan(atr[i]) or np.isnan(trend_sma[i]):
                continue

            if position == 0:
                if confirmed_long[i] and close[i] > trend_sma[i]:
                    position = 1
                    entry_price = float(close[i])
                    entry_bar = i
                    stop_price = entry_price - params.atr_trail_mult * float(atr[i])
                    breakeven_armed = False
                    raw_signal[i] = 1
                elif confirmed_short[i] and close[i] < trend_sma[i]:
                    position = -1
                    entry_price = float(close[i])
                    entry_bar = i
                    stop_price = entry_price + params.atr_trail_mult * float(atr[i])
                    breakeven_armed = False
                    raw_signal[i] = -1
            elif position == 1:
                hold_bars = i - entry_bar
                price_chg = (float(close[i]) - entry_price) / entry_price
                if not breakeven_armed and price_chg >= params.breakeven_trigger_pct:
                    if entry_price > stop_price:
                        stop_price = entry_price
                    breakeven_armed = True
                if breakeven_armed:
                    new_trail = float(close[i]) - params.atr_trail_mult * float(atr[i])
                    if new_trail > stop_price:
                        stop_price = new_trail
                stop_hit = float(low[i]) <= stop_price
                max_hold = hold_bars >= params.max_hold_bars
                if stop_hit or max_hold:
                    position = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
            elif position == -1:
                hold_bars = i - entry_bar
                price_chg = (entry_price - float(close[i])) / entry_price
                if not breakeven_armed and price_chg >= params.breakeven_trigger_pct:
                    if entry_price < stop_price:
                        stop_price = entry_price
                    breakeven_armed = True
                if breakeven_armed:
                    new_trail = float(close[i]) + params.atr_trail_mult * float(atr[i])
                    if new_trail < stop_price:
                        stop_price = new_trail
                stop_hit = float(high[i]) >= stop_price
                max_hold = hold_bars >= params.max_hold_bars
                if stop_hit or max_hold:
                    position = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = -1

        df = pd.DataFrame(index=data.index)
        df['signal'] = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')
