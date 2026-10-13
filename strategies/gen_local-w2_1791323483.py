from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureBreakoutParams:
    signature_window: int = 6
    breakout_window: int = 10
    area_threshold: float = 0.12
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureBreakoutParams]):
    strategy_id = "gen_local-w2_1791323483"

    @classmethod
    def params_type(cls) -> type[SignatureBreakoutParams]:
        return SignatureBreakoutParams

    @staticmethod
    def warmup_bars(params: SignatureBreakoutParams) -> int:
        return int(max(200, params.signature_window + 1, params.breakout_window + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureBreakoutParams) -> pd.DataFrame:
        if params.signature_window < 2 or params.breakout_window < 1:
            raise ValueError("Lookbacks must be positive; signature_window must be at least two.")
        if not np.isfinite(params.area_threshold) or not 0 <= params.area_threshold <= 0.5:
            raise ValueError("area_threshold must be between zero and 0.5.")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive.")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be one or two.")
        result = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        high = data["high"].where(data["high"] > 0)
        low = data["low"].where(data["low"] > 0)
        result["close_return"] = np.log(close).diff()
        result["log_range"] = np.log(high / low).where(high >= low)
        result["range_change"] = result["log_range"].diff()
        dx = result["close_return"]
        dy = result["range_change"]
        window = params.signature_window
        area = pd.Series(0.0, index=data.index)
        # Second-order signed area: half the sum of all ordered increment cross-products.
        for lag in range(1, window):
            cross = dx.shift(lag) * dy - dy.shift(lag) * dx
            area = area + 0.5 * cross.rolling(window - lag, min_periods=window - lag).sum()
        result["signature_area"] = area
        result["area_scale"] = dx.abs().rolling(window, min_periods=window).sum() * dy.abs().rolling(window, min_periods=window).sum()
        result["normalized_area"] = (area / result["area_scale"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["ma_200"] = close.rolling(200, min_periods=200).mean()
        result["prior_channel_high"] = high.rolling(params.breakout_window, min_periods=params.breakout_window).max().shift(1)
        result["entry"] = (
            close.gt(result["ma_200"])
            & close.gt(result["prior_channel_high"])
            & result["normalized_area"].gt(params.area_threshold)
        ).fillna(False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureBreakoutParams) -> SignalFrame:
        if params.max_hold_bars not in (1, 2) or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Use a positive profit target and a one- or two-bar holding limit.")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        holding = False
        held_bars = 0
        entry_price = 0.0
        for i in range(len(data)):
            if holding:
                # The preceding close's entry decision fills at this bar's open.
                if held_bars == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        holding = False
                        continue
                held_bars += 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or held_bars >= params.max_hold_bars:
                    holding = False
                    held_bars = 0
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                holding = True
                held_bars = 0
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = SignatureBreakoutParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    data.loc[index[1], "close"] = 101.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    assert result["size"].gt(0).all()
