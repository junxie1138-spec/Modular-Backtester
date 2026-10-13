from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBridgeParams:
    graph_window: int = 20
    atr_window: int = 14
    minimum_bridges: int = 2
    downside_share: float = 0.67
    shock_atr: float = 0.35
    entry_gap_atr: float = 0.20
    maximum_gap_atr: float = 1.20
    initial_stop_atr: float = 2.5
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GapBridgeParams]):
    strategy_id = 'gen_local-w2_1790994971'

    @classmethod
    def params_type(cls) -> type[GapBridgeParams]:
        return GapBridgeParams

    @staticmethod
    def warmup_bars(params: GapBridgeParams) -> int:
        return int(max(200, max(2, params.graph_window) + 1,
                       max(1, params.atr_window) + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapBridgeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr_window = max(1, int(params.atr_window))
        out['atr'] = true_range.rolling(atr_window, min_periods=atr_window).mean()
        out['previous_atr'] = out['atr'].shift(1)
        denominator = out['previous_atr'].where(out['previous_atr'] > 0)
        out['gap'] = ((data['open'] - previous_close) / denominator).replace(
            [np.inf, -np.inf], np.nan)
        out['gap_magnitude'] = out['gap'].abs()
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()

        # Horizontal visibility: every intermediate magnitude must be
        # strictly smaller than both endpoint magnitudes.
        magnitude = out['gap_magnitude']
        blocker = pd.Series(-np.inf, index=data.index, dtype=float)
        downside = pd.Series(0.0, index=data.index)
        shock_edges = pd.Series(0.0, index=data.index)
        window = max(2, int(params.graph_window))
        for lag in range(1, window + 1):
            historical = magnitude.shift(lag)
            ceiling = pd.Series(np.minimum(magnitude, historical), index=data.index)
            visible = (blocker < ceiling) & magnitude.notna() & historical.notna()
            historical_gap = out['gap'].shift(lag)
            if lag >= 3:
                shock = visible & (historical_gap.abs() >= params.shock_atr)
                shock_edges += shock.astype(float)
                downside += (shock & (historical_gap < 0)).astype(float)
            blocker = pd.Series(
                np.maximum(blocker.to_numpy(), historical.to_numpy()),
                index=data.index)

        complete = magnitude.rolling(window + 1, min_periods=window + 1).count().eq(window + 1)
        out['downside_bridges'] = downside.where(complete)
        out['shock_edges'] = shock_edges.where(complete)
        out['downside_share'] = (out['downside_bridges'] /
                                 out['shock_edges'].where(out['shock_edges'] > 0))
        out['entry_setup'] = (
            complete
            & (data['close'] > out['ma200'])
            & (data['close'] > data['open'])
            & (out['gap'] >= params.entry_gap_atr)
            & (out['gap'] <= params.maximum_gap_atr)
            & (out['downside_bridges'] >= params.minimum_bridges)
            & (out['downside_share'] >= params.downside_share)
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapBridgeParams) -> SignalFrame:
        n = len(data)
        desired = np.zeros(n, dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        previous_atr = indicators['previous_atr'].to_numpy(dtype=float)
        setups = indicators['entry_setup'].fillna(False).to_numpy(dtype=bool)
        active = False
        pending_entry = False
        armed = False
        entry = np.nan
        stop = np.nan
        peak = np.nan
        fill_bar = -1
        hold_limit = max(1, int(params.max_hold_bars))
        initial_multiple = max(0.01, float(params.initial_stop_atr))
        trail_multiple = max(0.01, float(params.trail_atr))
        arm_fraction = max(0.0, float(params.breakeven_pct))

        for i in range(n):
            if pending_entry:
                pending_entry = False
                if not (np.isfinite(opens[i]) and opens[i] > 0
                        and np.isfinite(previous_atr[i]) and previous_atr[i] > 0):
                    active = False
                else:
                    entry = opens[i]
                    stop = entry - initial_multiple * previous_atr[i]
                    peak = entry
                    fill_bar = i
                    armed = False

            if active:
                # An observed stop breach requests an exit at the next open.
                # A newly ratcheted stop applies only to subsequent bars.
                stop_hit = np.isfinite(lows[i]) and lows[i] <= stop
                expired = i - fill_bar + 1 >= hold_limit
                if stop_hit or expired:
                    active = False
                    armed = False
                    continue
                if np.isfinite(closes[i]):
                    peak = max(peak, closes[i])
                    if closes[i] >= entry * (1.0 + arm_fraction):
                        armed = True
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - trail_multiple * atr[i])
                    if closes[i] <= stop:
                        active = False
                        armed = False
                        continue
                desired[i] = 1
            elif setups[i] and i + 1 < n:
                active = True
                pending_entry = True
                desired[i] = 1

        df = pd.DataFrame({'signal': desired, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
