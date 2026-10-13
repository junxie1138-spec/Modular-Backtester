from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    flow_bars: int = 4
    rank_bars: int = 63
    vol_bars: int = 20
    entry_quantile: float = 0.80
    cancellation_quantile: float = 0.35
    target_vol: float = 0.12
    profit_target: float = 0.015
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = 'gen_local-w2_1791242606'

    @classmethod
    def params_type(cls) -> type[PressureParams]:
        return PressureParams

    @staticmethod
    def warmup_bars(params: PressureParams) -> int:
        return int(max(params.flow_bars + params.rank_bars + 1,
                       params.vol_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: PressureParams) -> pd.DataFrame:
        if (params.flow_bars < 2 or params.rank_bars < 2 or
                params.vol_bars < 2 or params.max_hold < 1 or
                not 0.0 < params.entry_quantile < 1.0 or
                not 0.0 < params.cancellation_quantile < 1.0 or
                not np.isfinite(params.target_vol) or params.target_vol <= 0.0 or
                not np.isfinite(params.profit_target) or params.profit_target <= 0.0):
            raise ValueError('Invalid strategy parameters')

        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0.0)
        opening = data['open'].where(data['open'] > 0.0)
        out['volume'] = data['volume'].where(data['volume'] >= 0.0)
        out['body_return'] = (close / opening - 1.0).replace([np.inf, -np.inf], np.nan)
        out['absolute_body'] = out['body_return'].abs()
        out['weighted_body'] = out['volume'] * out['body_return']
        out['weighted_absolute_body'] = out['volume'] * out['absolute_body']
        n = params.flow_bars
        out['body_sum'] = out['body_return'].rolling(n, min_periods=n).sum()
        out['absolute_body_sum'] = out['absolute_body'].rolling(n, min_periods=n).sum()
        out['weighted_body_sum'] = out['weighted_body'].rolling(n, min_periods=n).sum()
        out['weighted_absolute_body_sum'] = out['weighted_absolute_body'].rolling(n, min_periods=n).sum()
        out['price_balance'] = out['body_sum'] / out['absolute_body_sum'].replace(0.0, np.nan)
        out['volume_balance'] = out['weighted_body_sum'] / out['weighted_absolute_body_sum'].replace(0.0, np.nan)
        out['tilt'] = out['volume_balance'] - out['price_balance']
        out['absolute_tilt'] = out['tilt'].abs()
        out['cancellation'] = out['price_balance'].abs()
        r = params.rank_bars
        out['tilt_threshold'] = out['absolute_tilt'].shift(1).rolling(r, min_periods=r).quantile(params.entry_quantile)
        out['cancellation_threshold'] = out['cancellation'].shift(1).rolling(r, min_periods=r).quantile(params.cancellation_quantile)
        out['displacement'] = close.diff(n)
        out['close_return'] = close.pct_change(fill_method=None)
        out['annual_vol'] = out['close_return'].rolling(params.vol_bars, min_periods=params.vol_bars).std(ddof=0) * np.sqrt(252.0)
        out['size'] = (params.target_vol / out['annual_vol'].replace(0.0, np.nan)).clip(0.05, 0.95).fillna(0.05)
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        eligible = (
            (out['absolute_tilt'] > out['tilt_threshold']) &
            (out['cancellation'] <= out['cancellation_threshold']) &
            (out['tilt'] * out['volume_balance'] > 0.0) &
            (out['tilt'] * out['displacement'] > 0.0) &
            (out['annual_vol'] > 0.0)
        ).fillna(False)
        out['candidate'] = np.where(eligible, np.sign(out['tilt'].fillna(0.0)), 0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PressureParams) -> SignalFrame:
        count = len(data)
        decisions = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.05, dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        risk_sizes = indicators['size'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        held_size = 0.05

        for i in range(count):
            exited = False
            if position != 0:
                # A decision on the previous close enters at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        position = 0
                        exited = True
                if position != 0:
                    age = i - entry_bar + 1
                    gain = position * (closes[i] / entry_price - 1.0)
                    # Targets are observed at the close; the exit fills next open.
                    if ((np.isfinite(gain) and gain >= params.profit_target) or
                            age >= params.max_hold):
                        position = 0
                        exited = True

            if position == 0 and not exited and candidates[i] != 0:
                if (np.isfinite(closes[i]) and closes[i] > 0.0 and
                        np.isfinite(risk_sizes[i]) and risk_sizes[i] > 0.0):
                    position = int(candidates[i])
                    entry_bar = i + 1
                    entry_price = np.nan
                    held_size = float(np.clip(risk_sizes[i], 0.05, 0.95))

            decisions[i] = position
            sizes[i] = held_size

        df = pd.DataFrame({'signal': decisions, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({
        'open': 100.0, 'high': 103.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    data.loc[index[2], 'close'] = 102.0
    derived = pd.DataFrame({
        'candidate': [1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        'size': 0.40,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, PressureParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, -1, -1, 0, 0]
    assert result['size'].gt(0.0).all()
    flat_indicators = GeneratedStrategy.indicators(data, PressureParams())
    assert flat_indicators['candidate'].eq(0).all()
    assert flat_indicators['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
