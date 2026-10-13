from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEchoParams:
    lookback: int = 80
    min_correlation: float = 0.15
    min_score: float = 0.60
    atr_window: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 2.5
    max_hold: int = 10
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[GapEchoParams]):
    strategy_id = 'gen_local-w2_1791328582'

    @classmethod
    def params_type(cls) -> type[GapEchoParams]:
        return GapEchoParams

    def warmup_bars(self, params: GapEchoParams) -> int:
        return int(max(params.lookback + 6, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapEchoParams) -> pd.DataFrame:
        if params.lookback < 8 or params.atr_window < 1:
            raise ValueError('Invalid indicator windows')
        if not 0 < params.min_correlation < 1 or params.min_score <= 0:
            raise ValueError('Invalid forecast thresholds')
        if params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError('Invalid stop parameters')
        if params.max_hold < 1 or not 0 < params.min_size <= 1:
            raise ValueError('Invalid holding period or sizing floor')
        previous = data['close'].shift(1).where(lambda x: x > 0)
        gap = (data['open'].where(lambda x: x > 0) / previous - 1).replace([np.inf, -np.inf], np.nan)
        lagged = gap.shift(5)
        joint = gap.notna() & lagged.notna()
        y = gap.where(joint)
        x = lagged.where(joint)
        yr = y.rolling(params.lookback, min_periods=params.lookback)
        xr = x.rolling(params.lookback, min_periods=params.lookback)
        vy = yr.var(ddof=0)
        vx = xr.var(ddof=0)
        covariance = yr.cov(x, ddof=0)
        beta = covariance / vx.where(vx > 1e-12)
        correlation = (covariance / np.sqrt((vx * vy).where(lambda s: s > 1e-24))).clip(-1, 1)
        intercept = yr.mean() - beta * xr.mean()
        recent_gap_sum = gap.rolling(5, min_periods=5).sum()
        forecast = 5 * intercept + beta * recent_gap_sum
        residual_sigma = np.sqrt((vy * (1 - correlation.pow(2))).clip(lower=0))
        score = forecast / (np.sqrt(5.0) * residual_sigma).where(lambda s: s > 1e-8)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous.notna() & data[['high', 'low']].notna().all(axis=1))
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return pd.DataFrame({
            'gap': gap,
            'lagged_gap': lagged,
            'gap_variance': vy,
            'lagged_variance': vx,
            'covariance': covariance,
            'beta': beta,
            'correlation': correlation,
            'intercept': intercept,
            'recent_gap_sum': recent_gap_sum,
            'forecast': forecast,
            'residual_sigma': residual_sigma,
            'score': score,
            'true_range': true_range,
            'atr': atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapEchoParams) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        score = indicators['score'].to_numpy(dtype=float)
        correlation = indicators['correlation'].to_numpy(dtype=float)
        forecast = indicators['forecast'].to_numpy(dtype=float)
        valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
        eligible = (
            valid & np.isfinite(atr) & (atr > 0)
            & np.isfinite(score) & np.isfinite(correlation)
            & np.isfinite(forecast) & (forecast > 0)
            & (correlation >= params.min_correlation)
            & (score >= params.min_score)
        )
        active = False
        armed = False
        entry = peak = stop = 0.0
        weight = 1.0
        age = 0
        for i in range(n):
            # A prior close's entry decision becomes a position at this open.
            if not active and i > 0 and raw[i - 1] == 1:
                if not valid[i] or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    continue
                active = True
                armed = False
                age = 0
                entry = float(prices[i, 0])
                peak = entry
                stop = entry - params.trail_atr * atr[i - 1]
                weight = sizes[i - 1]
            if active:
                age += 1
                # Test the previously established stop before updating it.
                # ponytail: stop crossings exit next open; intraday execution needs a different contract.
                if not valid[i] or prices[i, 2] <= stop or age >= params.max_hold:
                    active = False
                    continue
                peak = max(peak, float(prices[i, 1]))
                armed = armed or peak >= entry * (1 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                raw[i] = 1
                sizes[i] = weight
            elif eligible[i]:
                raw[i] = 1
                confidence = score[i] / (score[i] + params.min_score)
                sizes[i] = float(np.clip(
                    params.min_size + (1 - params.min_size) * confidence,
                    params.min_size, 1.0,
                ))
        df = pd.DataFrame({'signal': raw, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100, 100, 102, 101, 100, 100],
        'high': [100, 102, 104, 102, 101, 101],
        'low': [99, 99, 101, 99, 99, 99],
        'close': [100, 102, 103, 100, 100, 100],
        'volume': [1000] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0] * 6,
        'score': [0.6, np.nan, np.nan, np.nan, np.nan, np.nan],
        'correlation': [0.5] * 6,
        'forecast': [0.01] * 6,
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, GapEchoParams(),
    ).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert np.allclose(result['size'].iloc[1:4], 0.625)
    assert result['size'].gt(0).all()
    assert result.index.equals(data.index)


if __name__ == '__main__':
    _self_check()
