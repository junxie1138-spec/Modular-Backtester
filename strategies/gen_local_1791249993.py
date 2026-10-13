from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    vol_window: int = 5
    rank_window: int = 126
    compression_quantile: float = 0.25
    shock_quantile: float = 0.10
    shock_window: int = 20
    cooldown_bars: int = 5
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local_1791249993'

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(
            params.vol_window + params.rank_window + 2,
            params.rank_window + max(params.shock_window, params.cooldown_bars) + 2,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None)
        out['volatility'] = out['return'].rolling(p.vol_window, min_periods=p.vol_window).std()
        out['compression_cutoff'] = out['volatility'].shift(1).rolling(p.rank_window, min_periods=p.rank_window).quantile(p.compression_quantile)
        out['shock_cutoff'] = out['return'].shift(1).rolling(p.rank_window, min_periods=p.rank_window).quantile(p.shock_quantile)
        out['downside_shock'] = ((out['return'] < out['shock_cutoff']) & (out['return'] < 0)).astype(float)
        out['recent_shocks'] = out['downside_shock'].rolling(p.shock_window, min_periods=p.shock_window).sum()
        out['cooldown_shocks'] = out['downside_shock'].rolling(p.cooldown_bars, min_periods=p.cooldown_bars).sum()
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_window, min_periods=p.atr_window).mean()
        out['stabilization_mean'] = close.rolling(p.vol_window, min_periods=p.vol_window).mean()
        required = ['return', 'volatility', 'compression_cutoff', 'shock_cutoff', 'recent_shocks', 'cooldown_shocks', 'atr', 'stabilization_mean']
        ready = (
            out[required].notna().all(axis=1)
            & (out['volatility'] <= out['compression_cutoff'])
            & (out['recent_shocks'] > 0)
            & (out['cooldown_shocks'] == 0)
            & (close > out['stabilization_mean'])
            & (out['atr'] > 0)
        ).fillna(False)
        out['entry'] = ready & ~ready.shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        pending_fill = False
        fixed_atr = np.nan
        stop_price = np.nan
        age = 0
        for i in range(len(data)):
            if active:
                if pending_fill:
                    # Entry is executed at this bar's open using the prior decision's ATR.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        pending_fill = False
                        continue
                    stop_price = opens[i] - params.stop_atr * fixed_atr
                    pending_fill = False
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop_price or age >= params.hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                pending_fill = True
                fixed_atr = atr[i]
                age = 0
                decisions[i] = 1
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'open': [110.0, 100.0, 98.0, 97.0, 95.0], 'close': [110.0, 100.0, 97.0, 95.0, 95.0]}, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False, False], 'atr': [2.0, 20.0, 0.1, 20.0, 2.0]}, index=index)
    strategy = GeneratedStrategy()
    stopped = strategy.generate_signals(data, indicators, None, CompressionParams()).data
    assert stopped['signal'].tolist() == [0, 1, 1, 1, 0]
    expired = strategy.generate_signals(data, indicators, None, CompressionParams(hold_bars=2)).data
    assert expired['signal'].tolist() == [0, 1, 1, 0, 0]
    assert (stopped['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
