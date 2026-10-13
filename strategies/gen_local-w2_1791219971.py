from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 12
    volatility_bars: int = 20
    area_threshold: float = 0.04
    target_volatility: float = 0.12
    profit_target: float = 0.025
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791219971"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(200, max(2, params.path_bars) + 2,
                       max(2, params.volatility_bars) + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        length = max(2, int(params.path_bars))
        volatility_length = max(2, int(params.volatility_bars))
        close = data['close'].where(data['close'] > 0)
        returns = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        range_fraction = ((data['high'] - data['low']) / close).where(
            data['high'] > data['low']
        )
        volatility_change = np.log(range_fraction).diff().replace(
            [np.inf, -np.inf], np.nan
        )
        dx = returns.fillna(0.0)
        dy = volatility_change.fillna(0.0)
        cumulative_x = dx.cumsum()
        cumulative_y = dy.cumsum()
        previous_x = cumulative_x.shift(1).fillna(0.0)
        previous_y = cumulative_y.shift(1).fillna(0.0)
        origin_x = cumulative_x.shift(length).fillna(0.0)
        origin_y = cumulative_y.shift(length).fillna(0.0)
        net_x = returns.rolling(length).sum()
        net_y = volatility_change.rolling(length).sum()
        cross = (previous_x * dy - previous_y * dx).rolling(length).sum()
        area = 0.5 * (cross - origin_x * net_y + origin_y * net_x)
        scale = (returns.abs().rolling(length).sum()
                 * volatility_change.abs().rolling(length).sum())
        out = pd.DataFrame(index=data.index)
        out['signature_area'] = (area / scale.where(scale > 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        out['net_return'] = net_x
        out['annual_volatility'] = returns.rolling(volatility_length).std() * np.sqrt(252.0)
        out['ma_200'] = close.rolling(200).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        area = indicators['signature_area']
        net_return = indicators['net_return']
        threshold = max(0.0, float(params.area_threshold))
        long_entry = (ready & (area < -threshold) & (net_return > 0)
                      & (data['close'] > indicators['ma_200']))
        short_entry = (ready & (area > threshold) & (net_return < 0)
                       & (data['close'] < indicators['ma_200']))
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        volatility = indicators['annual_volatility'].where(
            indicators['annual_volatility'] > 0
        )
        exposure = (max(0.001, float(params.target_volatility)) / volatility).clip(
            lower=0.05, upper=0.95
        ).fillna(0.05).to_numpy()
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.05, dtype=float)
        side = 0
        pending_side = 0
        entry_price = 0.0
        entry_bar = 0
        position_size = 0.05
        holding_bars = max(1, int(params.holding_bars))
        profit_target = max(0.0001, float(params.profit_target))
        for i in range(len(data)):
            if pending_side:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    side = pending_side
                    entry_price = opens[i]
                    entry_bar = i
                pending_side = 0
            if side:
                gain = side * (closes[i] / entry_price - 1.0)
                expired = i - entry_bar + 1 >= holding_bars
                if expired or (np.isfinite(gain) and gain >= profit_target):
                    side = 0
                else:
                    signals[i] = side
                sizes[i] = position_size
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending_side = int(candidates[i])
                position_size = float(exposure[i])
                signals[i] = pending_side
                sizes[i] = position_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
