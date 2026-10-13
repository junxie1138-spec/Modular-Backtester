from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    anchor_window: int = 20
    rank_window: int = 60
    atr_window: int = 14
    vol_window: int = 20
    release_percentile: float = 0.25
    target_vol: float = 0.12
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = 'gen_local-w2_1791259663'

    @classmethod
    def params_type(cls) -> type[ElasticParams]:
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        return int(max(params.anchor_window, params.atr_window + 1) + params.rank_window + params.vol_window + 1)

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        if min(params.anchor_window, params.rank_window, params.atr_window, params.vol_window) < 2:
            raise ValueError('Indicator windows must be at least two.')
        if not 0.0 < params.release_percentile < 1.0:
            raise ValueError('release_percentile must lie strictly between zero and one.')
        if params.target_vol <= 0.0 or params.stop_atr <= 0.0 or params.holding_bars < 1:
            raise ValueError('Risk parameters and holding_bars must be positive.')
        close = data['close'].where(data['close'] > 0.0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out['true_range'] = true_range
        out['atr'] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['anchor'] = close.rolling(params.anchor_window, min_periods=params.anchor_window).mean()
        out['stretch'] = (close - out['anchor']) / out['atr'].where(out['atr'] > 0.0)
        # pandas Rolling.rank(pct=True): https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.core.window.rolling.Rolling.rank.html
        out['percentile'] = out['stretch'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['previous_percentile'] = out['percentile'].shift(1)
        out['return'] = close.pct_change(fill_method=None)
        out['annual_vol'] = out['return'].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out['conviction'] = (-out['stretch'] / params.stop_atr).clip(lower=0.1, upper=1.0)
        out['risk_scale'] = (params.target_vol / out['annual_vol'].where(out['annual_vol'] > 0.0)).clip(lower=0.01, upper=1.0)
        out['entry'] = (
            (out['previous_percentile'] <= params.release_percentile)
            & (out['percentile'] > params.release_percentile)
            & (out['stretch'] < 0.0)
            & (out['stretch'] > out['stretch'].shift(1))
            & (out['atr'] > 0.0)
            & (out['annual_vol'] > 0.0)
        ).fillna(False).astype(bool)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ElasticParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.01, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        conviction = indicators['conviction'].to_numpy(dtype=float)
        risk_scale = indicators['risk_scale'].to_numpy(dtype=float)
        active = False
        pending = False
        stop_price = 0.0
        entry_conviction = 0.0
        age = 0
        for i in range(count):
            # A decision on the previous close fills at this bar's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    active = True
                    stop_price = opens[i] - params.stop_atr * atr[i - 1]
                    age = 0
            if active:
                age += 1
                if (not np.isfinite(closes[i])) or closes[i] < stop_price or age >= params.holding_bars:
                    active = False
                    continue
                if not np.isfinite(risk_scale[i]):
                    active = False
                    continue
                signals[i] = 1
                sizes[i] = float(np.clip(entry_conviction * risk_scale[i], 0.01, 1.0))
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(conviction[i]) and np.isfinite(risk_scale[i]):
                pending = True
                entry_conviction = float(conviction[i])
                signals[i] = 1
                sizes[i] = float(np.clip(entry_conviction * risk_scale[i], 0.01, 1.0))
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=12, freq='B')
    data = pd.DataFrame({'open': 102.0, 'high': 103.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'atr': 2.0, 'entry': False, 'conviction': 0.5, 'risk_scale': 0.8}, index=index)
    ind.loc[index[1], 'entry'] = True
    strategy = GeneratedStrategy()
    params = ElasticParams()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert np.isclose(result['size'].iloc[2], 0.4)
    data.loc[index[2], 'close'] = 97.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert result['size'].gt(0.0).all()
    ind['entry'] = False
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
