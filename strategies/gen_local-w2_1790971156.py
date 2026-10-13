from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    slope_bars: int = 5
    volatility_bars: int = 20
    shape_bars: int = 5
    history_bars: int = 180
    forecast_bars: int = 8
    percentile_bars: int = 90
    neighbor_percentile: float = 0.10
    conviction_percentile: float = 0.70
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1790971156'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.volatility_bars, params.slope_bars + 1)
                   + params.shape_bars + params.history_bars
                   + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if (min(p.slope_bars, p.volatility_bars, p.shape_bars,
                p.forecast_bars, p.percentile_bars, p.max_hold) < 2
                or p.history_bars < p.forecast_bars + p.shape_bars + 20
                or not 0 < p.neighbor_percentile < 1
                or not 0 < p.conviction_percentile < 1
                or not np.isfinite(p.profit_target)
                or p.profit_target <= 0):
            raise ValueError('Invalid analog strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['log_return'] = np.log(close).diff()
        out['volatility'] = out['log_return'].rolling(p.volatility_bars).std(ddof=0)
        out['latent_slope'] = out['log_return'].rolling(p.slope_bars).mean()
        out['innovation'] = (out['latent_slope'].diff()
                             / out['volatility'].replace(0.0, np.nan))
        features = []
        for lag in range(p.shape_bars):
            name = f'innovation_lag_{lag}'
            out[name] = out['innovation'].shift(lag)
            features.append(name)
        shape = out[features].to_numpy(dtype=float)
        lags = range(p.forecast_bars + p.shape_bars, p.history_bars + 1)
        distances = []
        outcomes = []
        for lag in lags:
            past = out[features].shift(lag).to_numpy(dtype=float)
            distances.append(np.mean((shape - past) ** 2, axis=1))
            # Each outcome ends at least shape_bars before today's decision.
            outcomes.append((close.shift(lag - p.forecast_bars)
                             / close.shift(lag) - 1.0).to_numpy(dtype=float))
        distance = np.column_stack(distances)
        outcome = np.column_stack(outcomes)
        ready = np.isfinite(distance).all(axis=1) & np.isfinite(outcome).all(axis=1)
        score = np.full(len(data), np.nan)
        radius = np.full(len(data), np.nan)
        count = np.zeros(len(data), dtype=int)
        if ready.any():
            d = distance[ready]
            radius[ready] = np.quantile(d, p.neighbor_percentile, axis=1)
            selected = d <= radius[ready, None]
            count[ready] = selected.sum(axis=1)
            score[ready] = np.where(selected, outcome[ready], 0.0).sum(axis=1) / count[ready]
        out['analog_radius'] = radius
        out['analog_count'] = count
        out['forecast'] = score
        out['conviction_threshold'] = (out['forecast'].abs()
                                       .rolling(p.percentile_bars)
                                       .quantile(p.conviction_percentile).shift(1))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        forecast = indicators['forecast'].to_numpy(dtype=float)
        threshold = indicators['conviction_threshold'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = (position * (closes[i] / entry_price - 1.0)
                        if valid_entry and np.isfinite(closes[i]) else np.nan)
                expired = i - entry_bar + 1 >= params.max_hold
                # Close-observed exits execute at the following open.
                if not valid_entry or expired or gain >= params.profit_target:
                    position = 0
                    entry_price = np.nan
                    continue
            elif (np.isfinite(forecast[i]) and np.isfinite(threshold[i])
                  and abs(forecast[i]) > threshold[i]):
                position = 1 if forecast[i] > 0 else -1
                entry_bar = i + 1
            desired[i] = position
        df = pd.DataFrame({'signal': desired, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    p = AnalogParams(max_hold=3)
    index = pd.date_range('2020-01-01', periods=8)
    for side in (1, -1):
        for target in (False, True):
            data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0,
                                 'close': 100.0, 'volume': 1000.0}, index=index)
            if target:
                data.loc[index[2], 'close'] = 100.0 * (1.0 + side * 0.03)
            indicators = pd.DataFrame({'forecast': float(side),
                                       'conviction_threshold': 0.5}, index=index)
            signals = strategy.generate_signals(data, indicators, None, p).data
            expected = [0, side, side, 0] if target else [0, side, side, side, 0]
            assert signals['signal'].iloc[:len(expected)].tolist() == expected
            assert signals['size'].gt(0).all()
    indicators[:] = np.nan
    assert strategy.generate_signals(data, indicators, None, p).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
