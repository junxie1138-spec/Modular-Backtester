from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDebtParams:
    depth_window: int = 63
    gap_window: int = 8
    atr_window: int = 14
    deep_atr: float = 3.0
    debt_atr: float = 0.8
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapDebtParams]):
    strategy_id = "gen_local_1790907243"

    @classmethod
    def params_type(cls):
        return GapDebtParams

    def warmup_bars(self, params):
        return int(max(params.depth_window, params.gap_window + 1, params.atr_window + 1) + 2)

    def indicators(self, data, params):
        if min(params.depth_window, params.gap_window, params.atr_window, params.hold_bars) < 1:
            raise ValueError("Window lengths and holding period must be positive.")
        if not all(np.isfinite(x) and x > 0 for x in (params.deep_atr, params.debt_atr, params.stop_atr)):
            raise ValueError("ATR thresholds must be finite and positive.")
        previous = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean().replace(0, np.nan)
        gap = data["open"] - previous
        change = data["close"] - previous
        # Debt records the part of each opening gap still unrepaired at that day's close.
        down = pd.concat([(-gap).clip(lower=0), (-change).clip(lower=0)], axis=1).min(axis=1, skipna=False)
        up = pd.concat([gap.clip(lower=0), change.clip(lower=0)], axis=1).min(axis=1, skipna=False)
        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["depth"] = (data["close"].rolling(params.depth_window).max() - data["close"]) / atr
        out["down_debt"] = down.rolling(params.gap_window).sum() / atr
        out["up_debt"] = up.rolling(params.gap_window).sum() / atr
        out["bull"] = (change > 0) & (data["close"] > data["open"])
        out["bear"] = (change < 0) & (data["close"] < data["open"])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        valid = indicators[["atr", "depth", "down_debt", "up_debt"]].notna().all(axis=1)
        long_setup = valid & (indicators["depth"] >= params.deep_atr) & (indicators["down_debt"] >= params.debt_atr) & indicators["bull"].fillna(False)
        short_setup = valid & (indicators["depth"] <= 1.0) & (indicators["up_debt"] >= params.debt_atr) & indicators["bear"].fillna(False)
        long_entry = (long_setup & long_setup.shift(1, fill_value=False)).to_numpy()
        short_entry = (short_setup & short_setup.shift(1, fill_value=False)).to_numpy()
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        entry_price = np.nan
        fixed_atr = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                stopped = not np.isfinite(entry_price) or not np.isfinite(close[i])
                if not stopped:
                    stopped = (position == 1 and close[i] < entry_price - params.stop_atr * fixed_atr) or (position == -1 and close[i] > entry_price + params.stop_atr * fixed_atr)
                if stopped or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
            elif i >= self.warmup_bars(params) - 1 and (long_entry[i] or short_entry[i]):
                position = 1 if long_entry[i] else -1
                entry_bar = i + 1
                entry_price = np.nan
                fixed_atr = atr[i]
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapDebtParams(depth_window=2, gap_window=2, atr_window=2)
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 93.0, "close": 100.0, "volume": 1000.0}, index=index)
    features = pd.DataFrame({"atr": 2.0, "depth": 4.0, "down_debt": 1.0, "up_debt": 0.0, "bull": False, "bear": False}, index=index)
    features.loc[index[4:6], "bull"] = True
    data.loc[index[6], "close"] = 94.0
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    features["depth"] = 0.5
    features["down_debt"] = 0.0
    features["up_debt"] = 1.0
    features["bull"] = False
    features.loc[index[4:6], "bear"] = True
    data.loc[index[6], "close"] = 106.0
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, -1, 0, 0, 0]
    features["bear"] = False
    assert (strategy.generate_signals(data, features, ctx, params).data["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
