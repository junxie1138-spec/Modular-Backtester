from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionCoreParams:
    core_window: int = 8
    compression_window: int = 5
    rank_window: int = 120
    regime_window: int = 200
    atr_window: int = 14
    compression_percentile: float = 0.25
    overlap_percentile: float = 0.75
    breakeven_gain: float = 0.018
    initial_stop_atr: float = 2.5
    trail_atr: float = 2.5
    max_hold_bars: int = 18
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[AuctionCoreParams]):
    strategy_id = "gen_local-w2_1791257464"

    @classmethod
    def params_type(cls) -> type[AuctionCoreParams]:
        return AuctionCoreParams

    @staticmethod
    def warmup_bars(params: AuctionCoreParams) -> int:
        return int(max(
            params.regime_window,
            params.atr_window + 1,
            params.compression_window + params.rank_window,
            params.core_window + params.rank_window,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AuctionCoreParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['ma200'] = close.rolling(max(200, params.regime_window)).mean()
        out['relative_range'] = out['true_range'] / close.replace(0, np.nan)
        out['compression'] = out['relative_range'].rolling(params.compression_window).mean()
        out['compression_rank'] = out['compression'].rolling(params.rank_window).rank(pct=True)
        out['core_upper'] = data['high'].rolling(params.core_window).min()
        out['core_lower'] = data['low'].rolling(params.core_window).max()
        out['union_upper'] = data['high'].rolling(params.core_window).max()
        out['union_lower'] = data['low'].rolling(params.core_window).min()
        out['overlap'] = (
            (out['core_upper'] - out['core_lower']).clip(lower=0)
            / (out['union_upper'] - out['union_lower']).replace(0, np.nan)
        )
        out['overlap_rank'] = out['overlap'].rolling(params.rank_window).rank(pct=True)
        out['setup'] = (
            (out['compression_rank'] <= params.compression_percentile)
            & (out['overlap_rank'] >= params.overlap_percentile)
            & (out['core_upper'] > out['core_lower'])
        )
        out['entry'] = (
            out['setup'].shift(1).fillna(False).astype(bool)
            & (previous_close <= out['core_upper'].shift(1))
            & (close > out['core_upper'].shift(1))
            & (close > out['ma200'])
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AuctionCoreParams,
    ) -> SignalFrame:
        # Decisions use closes; every position change executes at the next open.
        signals = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atrs = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        pending = False
        armed = False
        entry_price = 0.0
        entry_atr = 0.0
        stop = 0.0
        held = 0

        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    stop = entry_price - params.initial_stop_atr * entry_atr
                    held = 0
                    armed = False
                    active = True

            if active:
                held += 1
                close = closes[i]
                # Closing-price exits avoid assumptions about intrabar ordering.
                if not np.isfinite(close) or close <= 0:
                    signals[i] = 1
                    continue
                if close <= stop or held >= params.max_hold_bars:
                    active = False
                    continue
                if close >= entry_price * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, close - params.trail_atr * atrs[i])
                signals[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                pending = True
                entry_atr = atrs[i]
                signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = float(params.size)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = AuctionCoreParams()
    index = pd.date_range('2020-01-01', periods=7)
    close = np.array([100., 101., 103., 104., 102., 101., 100.])
    data = pd.DataFrame({
        'open': [100., 100., 101., 103., 104., 102., 101.],
        'high': close + 1,
        'low': close - 1,
        'close': close,
        'volume': 1000.,
    }, index=index)
    indicators = pd.DataFrame({
        'entry': [True, False, False, False, False, False, False],
        'atr': 1.,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 1, 0]
    assert (result.data['size'] > 0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
