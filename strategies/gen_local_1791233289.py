from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 16
    bridge_threshold: float = 0.45
    reset_threshold: float = 0.15
    reset_window: int = 5
    drawdown_window: int = 63
    min_drawdown: float = 0.015
    max_drawdown: float = 0.08
    recovery_bars: int = 3
    atr_window: int = 14
    initial_stop_atr: float = 2.5
    breakeven_gain: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local_1791233289'

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(200, params.graph_window + params.reset_window + 1,
                       params.drawdown_window, params.atr_window + 1,
                       params.recovery_bars + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if (params.graph_window < 5 or params.reset_window < 1
                or params.drawdown_window < 1 or params.atr_window < 1
                or params.recovery_bars < 1 or params.max_hold_bars < 1
                or not 0 <= params.reset_threshold < params.bridge_threshold <= 1
                or not 0 < params.min_drawdown < params.max_drawdown < 1
                or params.initial_stop_atr <= 0 or params.trail_atr <= 0
                or params.breakeven_gain <= 0):
            raise ValueError('Invalid strategy parameters')
        close = data['close'].astype(float)
        out = pd.DataFrame(index=data.index)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        peak = close.rolling(params.drawdown_window,
                             min_periods=params.drawdown_window).max()
        out['drawdown'] = close.div(peak.where(peak > 0)).sub(1.0)
        out['recovery'] = close.sub(close.shift(params.recovery_bars))
        previous = close.shift(1)
        true_range = pd.concat([
            data['high'].sub(data['low']),
            data['high'].sub(previous).abs(),
            data['low'].sub(previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = true_range.rolling(params.atr_window,
                                       min_periods=params.atr_window).mean()

        # An edge to lag j exists when every intermediate price is below
        # the straight line joining that price to today's closing price.
        minimum_slope = pd.Series(np.inf, index=data.index, dtype=float)
        bridges = pd.Series(0.0, index=data.index)
        for lag in range(1, params.graph_window + 1):
            slope = close.sub(close.shift(lag)).div(float(lag))
            visible = slope.lt(minimum_slope) & slope.notna()
            if lag >= 5:
                bridges = bridges.add(visible.astype(float))
            minimum_slope = pd.concat([minimum_slope, slope], axis=1).min(
                axis=1, skipna=False)
        complete = close.rolling(params.graph_window + 1,
                                 min_periods=params.graph_window + 1).count()
        out['bridge_density'] = bridges.div(float(params.graph_window - 4)).where(
            complete.eq(params.graph_window + 1))
        out['recent_density_floor'] = out['bridge_density'].shift(1).rolling(
            params.reset_window, min_periods=params.reset_window).min()
        out['entry'] = (
            out['bridge_density'].ge(params.bridge_threshold)
            & out['bridge_density'].shift(1).lt(params.bridge_threshold)
            & out['recent_density_floor'].le(params.reset_threshold)
            & out['drawdown'].between(-params.max_drawdown, -params.min_drawdown)
            & close.gt(out['ma200'])
            & out['recovery'].gt(0)
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        regime = indicators['ma200'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry_price = 0.0
        entry_bar = 0
        peak_close = 0.0
        stop = 0.0

        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    active = False
                    continue
                entry_price = opens[i]
                entry_bar = i
                peak_close = entry_price
                stop = entry_price - params.initial_stop_atr * atr[i - 1]
                armed = False

            if active:
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    active = False
                    continue
                # Stops are evaluated at the close; exits fill next open.
                if (closes[i] <= stop or closes[i] < regime[i]
                        or i - entry_bar + 1 >= params.max_hold_bars):
                    active = False
                    continue
                peak_close = max(peak_close, closes[i])
                if closes[i] >= entry_price * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, peak_close - params.trail_atr * atr[i])
                if closes[i] <= stop:
                    active = False
                    continue
                signals[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                pending = True
                signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['size'] = 1.0
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': [99.0, 100.0, 101.0, 103.0, 104.0, 101.0, 101.0],
        'close': [99.0, 101.0, 103.0, 104.0, 101.0, 101.0, 101.0],
    }, index=index)
    indicators = pd.DataFrame({
        'atr': 1.0, 'ma200': 90.0,
        'entry': [True, False, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, VisibilityParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result['size'].gt(0).all()
    time_exit = GeneratedStrategy.generate_signals(
        None, data, indicators, None,
        VisibilityParams(max_hold_bars=2)).data
    assert time_exit['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
