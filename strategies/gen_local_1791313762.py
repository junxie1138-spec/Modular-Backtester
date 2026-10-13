from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    phase_bars: int = 5
    rank_window: int = 126
    history_bars: int = 504
    holding_bars: int = 18
    atr_window: int = 14
    min_analogs: int = 12
    forecast_threshold: float = 0.5
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1791313762'

    @classmethod
    def params_type(cls):
        return ShapeParams

    @staticmethod
    def warmup_bars(params: ShapeParams) -> int:
        return int(max(
            params.history_bars + params.rank_window
            + 3 * params.phase_bars + params.holding_bars + 1,
            params.atr_window + params.holding_bars
            + params.history_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        integer_values = (
            params.phase_bars, params.rank_window, params.history_bars,
            params.holding_bars, params.atr_window, params.min_analogs,
        )
        if any(v < 1 or int(v) != v for v in integer_values):
            raise ValueError('Lookbacks and minimum analog count must be positive integers')
        if params.min_analogs > params.history_bars:
            raise ValueError('Minimum analog count exceeds history window')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')
        if not np.isfinite(params.forecast_threshold) or params.forecast_threshold <= 0:
            raise ValueError('forecast_threshold must be finite and positive')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().where(lambda s: s > 0)
        out['phase_return'] = close.pct_change(
            periods=params.phase_bars, fill_method=None
        ).replace([np.inf, -np.inf], np.nan)
        out['phase_rank'] = out['phase_return'].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out['old_rank'] = out['phase_rank'].shift(2 * params.phase_bars)
        out['middle_rank'] = out['phase_rank'].shift(params.phase_bars)
        out['old_bucket'] = np.floor(out['old_rank'] * 3).clip(upper=2)
        out['middle_bucket'] = np.floor(out['middle_rank'] * 3).clip(upper=2)
        out['new_bucket'] = np.floor(out['phase_rank'] * 3).clip(upper=2)
        out['shape'] = (
            9 * out['old_bucket'] + 3 * out['middle_bucket']
            + out['new_bucket']
        )

        # Labels become available only after the entire holding horizon has elapsed.
        out['matured_shape'] = out['shape'].shift(params.holding_bars)
        out['realized_atr_move'] = (
            (close - close.shift(params.holding_bars))
            / out['atr'].shift(params.holding_bars)
        ).replace([np.inf, -np.inf], np.nan)
        valid = out['matured_shape'].notna() & out['realized_atr_move'].notna()
        out['history_ready'] = valid.astype(float).rolling(
            params.history_bars, min_periods=params.history_bars
        ).sum().eq(params.history_bars)
        out['analog_count'] = 0.0
        out['forecast'] = np.nan
        for code in range(27):
            member = valid & out['matured_shape'].eq(code)
            count = member.astype(float).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            total = out['realized_atr_move'].where(member, 0.0).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            current = out['shape'].eq(code)
            out.loc[current, 'analog_count'] = count.loc[current]
            out.loc[current, 'forecast'] = (
                total / count.where(count >= params.min_analogs)
            ).loc[current]

        eligible = (
            out['history_ready'] & out['atr'].notna()
            & out['forecast'].notna() & close.notna()
        )
        eligible &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        out['candidate'] = np.select([
            eligible & out['forecast'].ge(params.forecast_threshold),
            eligible & out['forecast'].le(-params.forecast_threshold),
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ShapeParams,
    ) -> SignalFrame:
        candidate = indicators['candidate'].fillna(0).astype(int)
        confirmed = candidate.where(
            candidate.eq(candidate.shift(1)), 0
        ).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan

        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    # The prior close's decision fills at this session's open.
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                if position != 0 and np.isfinite(closes[i]):
                    stopped = (
                        closes[i] < entry_price - params.stop_atr * entry_atr
                        if position == 1 else
                        closes[i] > entry_price + params.stop_atr * entry_atr
                    )
                    expired = i - entry_bar + 1 >= params.holding_bars
                    if stopped or expired:
                        position = 0
            elif (
                confirmed[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(closes[i]) and closes[i] > 0
            ):
                position = int(confirmed[i])
                entry_bar = i + 1
                entry_price = np.nan
                entry_atr = atr[i]
            targets[i] = position

        df = pd.DataFrame({'signal': targets, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=10, freq='B')
    data = pd.DataFrame({
        'open': [100, 100, 105, 110, 103, 100, 100, 95, 94, 100],
        'close': [100, 100, 110, 103, 100, 100, 100, 94, 100, 100],
    }, index=index)
    indicators = pd.DataFrame({
        'candidate': [1, 1, 0, 0, 0, -1, -1, 0, 0, 0],
        'atr': 2.0,
    }, index=index)
    params = ShapeParams(stop_atr=2.0)
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, params
    ).data
    # Confirms delayed fills, both directions, and a fixed rather than trailing stop.
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0, -1, -1, 0]
    params.holding_bars = 2
    timed = GeneratedStrategy.generate_signals(
        None, data, indicators, None, params
    ).data
    assert timed['signal'].iloc[4] == 0
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
