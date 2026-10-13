from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 20
    reconnect_span: int = 6
    bottleneck_span: int = 2
    absorption_window: int = 3
    absorption_threshold: float = 0.20
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local-w2_1790996016'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(params.graph_window + 1, params.absorption_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        volume = data['volume'].astype(float).clip(lower=0)
        previous = close.shift(1)
        tr = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
        out['atr'] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['price_mean'] = close.rolling(params.graph_window, min_periods=params.graph_window).mean()

        # Horizontal visibility requires every intermediate close to lie
        # strictly below both endpoint closes. Adjacent bars always connect.
        span = pd.Series(1.0, index=data.index).where(close.notna() & previous.notna())
        intermediate_max = previous.copy()
        for lag in range(2, params.graph_window):
            anchor = close.shift(lag)
            visible = (close > intermediate_max) & (anchor > intermediate_max)
            span = span.where(~visible, float(lag))
            intermediate_max = pd.concat([intermediate_max, anchor], axis=1).max(axis=1, skipna=False)
        complete = close.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        out['visibility_span'] = span.where(complete)
        out['previous_span'] = out['visibility_span'].shift(1)

        day_range = (high - low).replace(0.0, np.nan)
        location = ((2.0 * close - high - low) / day_range).clip(-1.0, 1.0)
        location = location.mask((high == low) & close.notna(), 0.0)
        out['closing_location'] = location
        out['weighted_location'] = location * volume
        out['absorption'] = out['weighted_location'].rolling(params.absorption_window, min_periods=params.absorption_window).sum() / volume.rolling(params.absorption_window, min_periods=params.absorption_window).sum().replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        topology = (
            indicators['previous_span'].le(params.bottleneck_span)
            & indicators['visibility_span'].ge(params.reconnect_span)
            & data['close'].lt(indicators['price_mean'])
        )
        absorption = indicators['absorption'].ge(params.absorption_threshold)
        ready = indicators[['atr', 'price_mean', 'visibility_span', 'previous_span', 'absorption']].notna().all(axis=1)
        entries = (topology & absorption & ready & indicators['atr'].gt(0)).fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        entry_bar = -1
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and np.isfinite(atr[i - 1]):
                    active = True
                    entry_bar = i
                    # Freeze the stop at the next-bar open using entry-decision ATR.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                else:
                    signals[i] = 0
                    continue

            if active:
                stop_hit = np.isfinite(closes[i]) and closes[i] < stop
                time_exit = i - entry_bar + 1 >= params.hold_bars
                if stop_hit or time_exit:
                    active = False
                    stop = np.nan
                else:
                    signals[i] = 1
            elif i >= warmup - 1 and entries[i]:
                pending = True
                signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
