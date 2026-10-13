from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 4
    volume_window: int = 20
    percentile_window: int = 126
    entry_percentile: float = 0.80
    volatility_window: int = 20
    atr_window: int = 14
    target_volatility: float = 0.12
    stop_atr: float = 1.5
    holding_bars: int = 2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1790994850"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(
            max(1, params.volume_window) + max(2, params.path_window)
            + max(2, params.percentile_window) + 2,
            max(2, params.volatility_window) + 1,
            max(1, params.atr_window) + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        length = max(2, int(params.path_window))
        volume_window = max(1, int(params.volume_window))
        percentile_window = max(2, int(params.percentile_window))
        volatility_window = max(2, int(params.volatility_window))
        atr_window = max(1, int(params.atr_window))
        previous_close = data['close'].shift(1)
        out['return'] = (data['close'] / previous_close.where(previous_close > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out['range'] = (data['high'] - data['low']).clip(lower=0.0)
        out['body_efficiency'] = ((data['close'] - data['open']) / out['range'].replace(0.0, np.nan)).clip(-1.0, 1.0)
        out['volume_baseline'] = data['volume'].shift(1).rolling(volume_window, min_periods=volume_window).mean()
        out['relative_volume'] = (data['volume'] / out['volume_baseline'].where(out['volume_baseline'] > 0)).clip(0.0, 5.0)
        out['participation'] = out['body_efficiency'] * out['relative_volume']

        # Half the sum of ordered cross products is the antisymmetric
        # second-order signature of the participation/return path.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, length):
            pair = (out['participation'].shift(lag) * out['return']
                    - out['return'].shift(lag) * out['participation'])
            area = area + pair.rolling(length - lag, min_periods=length - lag).sum()
        out['signature_area'] = 0.5 * area
        out['participation_variation'] = out['participation'].abs().rolling(length, min_periods=length).sum()
        out['return_variation'] = out['return'].abs().rolling(length, min_periods=length).sum()
        out['area_scale'] = out['participation_variation'] * out['return_variation']
        out['normalized_area'] = out['signature_area'] / out['area_scale'].where(out['area_scale'] > 0)
        out['area_threshold'] = out['normalized_area'].shift(1).rolling(percentile_window, min_periods=percentile_window).quantile(float(np.clip(params.entry_percentile, 0.01, 0.99)))
        out['true_range'] = pd.concat([
            out['range'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(atr_window, min_periods=atr_window).mean()
        out['realized_volatility'] = out['return'].rolling(volatility_window, min_periods=volatility_window).std(ddof=0) * np.sqrt(252.0)
        cap = float(np.clip(params.max_size, 0.01, 1.0))
        target = max(0.0001, float(params.target_volatility))
        out['target_size'] = (target / out['realized_volatility'].where(out['realized_volatility'] > 0)).clip(lower=0.01, upper=cap)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        cap = float(np.clip(params.max_size, 0.01, 1.0))
        signals = np.zeros(len(data), dtype=np.int64)
        sizes = np.full(len(data), cap, dtype=float)
        eligible = (
            (indicators['normalized_area'] > indicators['area_threshold'])
            & (indicators['normalized_area'] > 0.0)
            & (indicators['return'] > 0.0)
            & (indicators['body_efficiency'] > 0.0)
            & (indicators['atr'] > 0.0)
            & indicators['target_size'].notna()
        ).fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        targets = indicators['target_size'].to_numpy(dtype=float)
        holding_bars = int(np.clip(params.holding_bars, 1, 2))
        stop_multiple = max(0.01, float(params.stop_atr))
        active = False
        pending_entry = False
        frozen_atr = 0.0
        frozen_size = cap
        stop_price = 0.0
        age = 0

        for i in range(len(data)):
            # A decision at the preceding close fills at this session's open.
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    stop_price = opens[i] - stop_multiple * frozen_atr
                    active = True
                    age = 0
            if active:
                age += 1
                sizes[i] = frozen_size
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                if stopped or age >= holding_bars:
                    active = False
                    signals[i] = 0
                else:
                    signals[i] = 1
            elif eligible[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                frozen_atr = float(atrs[i])
                frozen_size = float(targets[i])
                pending_entry = True
                signals[i] = 1
                sizes[i] = frozen_size

        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(cap).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
