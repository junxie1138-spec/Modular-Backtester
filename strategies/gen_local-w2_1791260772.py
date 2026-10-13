from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecruitmentParams:
    ma_window: int = 20
    recruitment_window: int = 8
    recruitment_lag: int = 4
    z_entry: float = 0.5
    z_cap: float = 1.75
    recruitment_step: float = 0.125
    profit_target: float = 0.006
    max_hold_bars: int = 2

    def __post_init__(self):
        if self.ma_window < 2 or self.recruitment_window < 2 or self.recruitment_lag < 1:
            raise ValueError('Invalid lookback')
        if not (0 < self.z_entry < self.z_cap < np.inf):
            raise ValueError('Invalid z-score thresholds')
        if not (0 < self.recruitment_step <= 1):
            raise ValueError('Invalid recruitment threshold')
        if not (0 < self.profit_target < np.inf) or self.max_hold_bars not in (1, 2):
            raise ValueError('Invalid exit parameters')


class GeneratedStrategy(BaseStrategy[RecruitmentParams]):
    strategy_id = 'gen_local-w2_1791260772'

    @classmethod
    def params_type(cls):
        return RecruitmentParams

    def warmup_bars(self, params: RecruitmentParams) -> int:
        return int(max(params.ma_window, params.recruitment_window + params.recruitment_lag))

    def indicators(self, data: pd.DataFrame, params: RecruitmentParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        opening = data['open'].astype(float)
        valid = np.isfinite(close) & np.isfinite(opening) & (close > 0) & (opening > 0)
        clean_close = close.where(np.isfinite(close) & (close > 0))
        out['ma'] = clean_close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['price_std'] = clean_close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['z_score'] = (clean_close - out['ma']) / out['price_std'].replace(0.0, np.nan)
        out['candle_vote'] = (0.5 + 0.5 * np.sign(close - opening)).where(valid)
        out['participation'] = out['candle_vote'].rolling(
            params.recruitment_window, min_periods=params.recruitment_window
        ).mean()
        out['participation_change'] = out['participation'] - out['participation'].shift(params.recruitment_lag)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecruitmentParams,
    ) -> SignalFrame:
        z = indicators['z_score']
        participation = indicators['participation']
        change = indicators['participation_change']
        long_entry = (
            z.between(params.z_entry, params.z_cap)
            & (participation >= 0.625)
            & (change >= params.recruitment_step)
        )
        short_entry = (
            z.between(-params.z_cap, -params.z_entry)
            & (participation <= 0.375)
            & (change <= -params.recruitment_step)
        )
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        opening = data['open'].to_numpy(dtype=float)
        closing = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        held_bars = 0
        entry_price = np.nan

        for i in range(len(data)):
            if position != 0:
                # The previous close's entry decision fills at this bar's open.
                if held_bars == 0:
                    entry_price = opening[i]
                held_bars += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closing[i]) and closing[i] > 0
                gain = position * (closing[i] / entry_price - 1.0) if valid_entry and valid_close else np.nan
                # Targets are observed at the close; exits fill next bar.
                if (
                    not valid_entry
                    or not valid_close
                    or gain >= params.profit_target
                    or held_bars >= params.max_hold_bars
                ):
                    position = 0
                    held_bars = 0
                    entry_price = np.nan
            elif entries[i] != 0 and np.isfinite(closing[i]) and closing[i] > 0:
                position = int(entries[i])
                held_bars = 0
            decisions[i] = position

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = RecruitmentParams()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    signals = pd.DataFrame({
        'z_score': [0.75, np.nan, np.nan, np.nan, np.nan],
        'participation': [0.75, np.nan, np.nan, np.nan, np.nan],
        'participation_change': [0.25, np.nan, np.nan, np.nan, np.nan],
    }, index=index)
    result = strategy.generate_signals(data, signals, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], 'close'] = 101.0
    assert strategy.generate_signals(data, signals, None, params).data['signal'].tolist() == [0, 1, 0, 0, 0]
    signals.loc[index[0], ['z_score', 'participation', 'participation_change']] = [-0.75, 0.25, -0.25]
    data.loc[index[1], 'close'] = 99.0
    assert strategy.generate_signals(data, signals, None, params).data['signal'].tolist() == [0, -1, 0, 0, 0]
    signals.loc[index[0], 'participation_change'] = 0.0
    assert strategy.generate_signals(data, signals, None, params).data['signal'].eq(0).all()
    flat = pd.concat([data.iloc[[0]]] * 30, ignore_index=True)
    flat.index = pd.date_range('2020-01-01', periods=len(flat))
    derived = strategy.indicators(flat, params)
    assert derived['z_score'].isna().all()
    assert strategy.generate_signals(flat, derived, None, params).data['signal'].eq(0).all()
    assert strategy.warmup_bars(params) >= max(params.ma_window, params.recruitment_window + params.recruitment_lag)
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
