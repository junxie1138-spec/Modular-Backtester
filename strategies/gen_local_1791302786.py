from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 24
    history_bars: int = 252
    match_radius: float = 0.25
    min_analogs: int = 8
    forecast_threshold: float = 0.20
    holding_bars: int = 4
    atr_period: int = 14
    stop_atr: float = 1.75


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791302786'

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.history_bars + max(200, params.pattern_window + 4, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.pattern_window >= 4 and params.history_bars > params.pattern_window + params.holding_bars and 3 <= params.holding_bars <= 5 and params.atr_period >= 1 and params.match_radius > 0 and params.min_analogs >= 1 and 0 < params.forecast_threshold < 1 and params.stop_atr > 0):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = returns.shift(2), returns.shift(1), returns
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns retain chronological order, giving deterministic motifs.
        motif = (4 * (a > b).astype(int) + 2 * (a > c).astype(int) + (b > c).astype(int)).astype(float).where(valid)
        out['motif'] = motif
        codes = (0, 1, 3, 4, 6, 7)
        transition_valid = motif.notna() & motif.shift(1).notna()
        joint_entropy = pd.Series(0.0, index=data.index)
        source_entropy = pd.Series(0.0, index=data.index)
        feature_names = []
        for code in codes:
            name = f'motif_frequency_{code}'
            out[name] = motif.eq(code).astype(float).where(valid).rolling(params.pattern_window).mean()
            feature_names.append(name)
            source_probability = motif.shift(1).eq(code).astype(float).where(transition_valid).rolling(params.pattern_window).mean()
            out[f'source_frequency_{code}'] = source_probability
            source_entropy -= source_probability * np.log(source_probability.clip(lower=1e-12))
            for following in codes:
                probability = (motif.shift(1).eq(code) & motif.eq(following)).astype(float).where(transition_valid).rolling(params.pattern_window).mean()
                out[f'transition_{code}_{following}'] = probability
                joint_entropy -= probability * np.log(probability.clip(lower=1e-12))
        out['complexity'] = ((joint_entropy - source_entropy) / np.log(6.0)).clip(0.0, 1.0)
        feature_names.append('complexity')
        out['ma200'] = close.rolling(200).mean()
        out['regime'] = pd.Series(np.where(close >= out['ma200'], 1.0, -1.0), index=data.index).where(close.notna() & out['ma200'].notna())
        previous_close = close.shift(1)
        true_range = pd.concat([(data['high'] - data['low']).abs(), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = true_range.rolling(params.atr_period).mean()
        out['atr_fraction'] = (out['atr'] / close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        features = out[feature_names].to_numpy(dtype=float)
        n = len(data)
        numerator = np.zeros(n, dtype=float)
        denominator = np.zeros(n, dtype=float)
        absolute_outcomes = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=int)
        current_motif = motif.to_numpy()
        regime = out['regime'].to_numpy()
        price = close.to_numpy()
        atr_fraction = out['atr_fraction'].to_numpy()
        first_lag = params.pattern_window + params.holding_bars + 1
        # Each historical outcome ends before the current pattern window starts.
        for lag in range(first_lag, min(params.history_bars, n - 1) + 1):
            distance = np.sqrt(np.square(features[lag:] - features[:-lag]).sum(axis=1))
            outcomes = (price[params.holding_bars:n-lag+params.holding_bars] / np.where(price[:-lag] > 0, price[:-lag], np.nan) - 1.0) / np.where(atr_fraction[:-lag] > 0, atr_fraction[:-lag], np.nan)
            eligible = np.isfinite(distance) & np.isfinite(outcomes) & (distance <= params.match_radius) & (current_motif[lag:] == current_motif[:-lag]) & (regime[lag:] == regime[:-lag])
            weights = np.where(eligible, np.exp(-0.5 * np.square(distance / params.match_radius)), 0.0)
            safe_outcomes = np.where(eligible, outcomes, 0.0)
            numerator[lag:] += weights * safe_outcomes
            absolute_outcomes[lag:] += weights * np.abs(safe_outcomes)
            denominator[lag:] += weights
            counts[lag:] += eligible.astype(int)
        out['analog_count'] = counts
        out['forecast_atr'] = np.divide(numerator, denominator, out=np.full(n, np.nan), where=denominator > 0)
        out['forecast_strength'] = np.divide(numerator, absolute_outcomes, out=np.full(n, np.nan), where=absolute_outcomes > 0)
        ready = out['analog_count'].ge(params.min_analogs) & out['atr'].gt(0) & out['ma200'].notna()
        long_entry = ready & out['regime'].eq(1) & out['forecast_strength'].ge(params.forecast_threshold)
        short_entry = ready & out['regime'].eq(-1) & out['forecast_strength'].le(-params.forecast_threshold)
        out['entry_direction'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        desired = np.zeros(n, dtype=int)
        entries = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        stop_distance = np.nan
        for i in range(n):
            if position:
                if i == entry_bar:
                    # The preceding close's decision fills at this bar's open.
                    entry_price = opens[i]
                    stop_distance = params.stop_atr * atr[i - 1]
                invalid = not (np.isfinite(entry_price) and entry_price > 0 and np.isfinite(stop_distance) and stop_distance > 0 and np.isfinite(closes[i]))
                stopped = (position == 1 and closes[i] < entry_price - stop_distance) or (position == -1 and closes[i] > entry_price + stop_distance)
                expired = i - entry_bar + 1 >= params.holding_bars
                if invalid or stopped or expired:
                    position = 0
                desired[i] = position
                continue
            if entries[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
                stop_distance = np.nan
            desired[i] = position
        df = pd.DataFrame({'signal': desired, 'size': np.full(n, 1.0)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 112.0, 'low': 90.0, 'close': [100.0, 100.0, 110.0, 97.0, 96.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry_direction': [1, 0, 0, 0, 0, 0, 0, 0], 'atr': 2.0}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params(holding_bars=5)).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert result['size'].gt(0).all()
    indicators['entry_direction'] = [-1, 0, 0, 0, 0, 0, 0, 0]
    data['close'] = [100.0, 100.0, 90.0, 103.0, 104.0, 100.0, 100.0, 100.0]
    result = strategy.generate_signals(data, indicators, None, Params(holding_bars=5)).data
    assert result['signal'].tolist() == [0, -1, -1, -1, -1, 0, 0, 0]
    indicators['entry_direction'] = [1, 0, 0, 0, 0, 0, 0, 0]
    data['close'] = 100.0
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
