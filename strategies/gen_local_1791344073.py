from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRecoveryParams:
    peak_window: int = 60
    range_window: int = 4
    atr_window: int = 14
    minimum_drawdown: float = 0.03
    maximum_downside_share: float = 0.15
    minimum_close_location: float = 0.60
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RangeRecoveryParams]):
    strategy_id = "gen_local_1791344073"

    @classmethod
    def params_type(cls) -> type[RangeRecoveryParams]:
        return RangeRecoveryParams

    def warmup_bars(self, params: RangeRecoveryParams) -> int:
        return int(max(params.peak_window, params.atr_window + 1, params.range_window + 2))

    def indicators(self, data: pd.DataFrame, params: RangeRecoveryParams) -> pd.DataFrame:
        if min(params.peak_window, params.range_window, params.atr_window, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not (0 < params.minimum_drawdown < 1 and 0 <= params.maximum_downside_share <= 1 and 0 <= params.minimum_close_location <= 1 and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Invalid entry thresholds or stop multiplier")
        out = pd.DataFrame(index=data.index)
        high, low, close = data['high'], data['low'], data['close']
        out['range'] = (high - low).where(high > low)
        previous_low, previous_high = low.shift(1), high.shift(1)
        lower_boundary = pd.concat([high, previous_low], axis=1).min(axis=1, skipna=False)
        upper_boundary = pd.concat([low, previous_high], axis=1).max(axis=1, skipna=False)
        out['downside_share'] = ((lower_boundary - low).clip(lower=0) / out['range']).clip(0, 1)
        out['upside_share'] = ((high - upper_boundary).clip(lower=0) / out['range']).clip(0, 1)
        out['downside_mean'] = out['downside_share'].rolling(params.range_window, min_periods=params.range_window).mean()
        out['upside_mean'] = out['upside_share'].rolling(params.range_window, min_periods=params.range_window).mean()
        out['exploration_balance'] = out['upside_mean'] - out['downside_mean']
        out['downside_change'] = out['downside_mean'].diff()
        out['close_location'] = (close - low) / out['range']
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = close / out['peak'].where(out['peak'] > 0) - 1.0
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out['ready'] = (
            (out['drawdown'] <= -params.minimum_drawdown)
            & (out['exploration_balance'] > 0)
            & (out['downside_mean'] <= params.maximum_downside_share)
            & (out['downside_change'] <= 0)
            & (out['close_location'] >= params.minimum_close_location)
            & (out['atr'] > 0)
        ).fillna(False)
        out['confirmed'] = out['ready'] & out['ready'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeRecoveryParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        first_decision = self.warmup_bars(params) - 1
        pending = False
        pending_atr = 0.0
        entry_bar = -1
        stop_price = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_bar = i
                    # Fix the stop once, using next-open entry and decision-bar ATR.
                    stop_price = opens[i] - params.stop_atr * pending_atr
            if entry_bar >= 0:
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    entry_bar = -1
                    stop_price = np.nan
                    continue
                signals[i] = 1
            elif i >= first_decision and confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                signals[i] = 1
                pending = True
                pending_atr = atr[i]
        df['signal'] = signals
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RangeRecoveryParams(peak_window=2, range_window=2, atr_window=2, hold_bars=3)
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived['confirmed'].any()
    ready = pd.Series([False, False, False, True, True, False, False, False, False, False], index=index)
    indicators = pd.DataFrame({'atr': 1.0, 'confirmed': ready & ready.shift(1, fill_value=False)}, index=index)
    held = strategy.generate_signals(data, indicators, None, params).data
    assert held['signal'].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    stopped_data = data.copy()
    stopped_data.loc[index[5], 'close'] = 97.0
    stopped = strategy.generate_signals(stopped_data, indicators, None, params).data
    assert stopped['signal'].tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert (stopped['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
