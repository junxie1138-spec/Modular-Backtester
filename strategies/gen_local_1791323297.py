from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_bars: int = 63
    debt_bars: int = 20
    breakout_bars: int = 5
    atr_bars: int = 14
    min_depth: float = 0.04
    stop_atr: float = 2.5
    holding_bars: int = 18
    min_size: float = 0.25
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1791323297"

    @classmethod
    def params_type(cls):
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(params.peak_bars + max(params.debt_bars, params.breakout_bars, params.atr_bars) + 2)

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if min(params.peak_bars, params.debt_bars, params.breakout_bars, params.atr_bars, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding_bars must be positive.")
        if not (0.0 < params.min_depth < 1.0 and 0.0 < params.min_size <= 1.0):
            raise ValueError("Invalid min_depth or min_size.")
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0.0):
            raise ValueError("stop_atr must be finite and positive.")
        if not (np.isfinite(params.entry_slippage_bps) and params.entry_slippage_bps >= 0.0):
            raise ValueError("entry_slippage_bps must be finite and nonnegative.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["peak"] = close.rolling(params.peak_bars, min_periods=params.peak_bars).max()
        out["depth"] = (1.0 - close / out["peak"].where(out["peak"] > 0.0)).clip(lower=0.0)
        out["underwater_volume"] = data["volume"].clip(lower=0.0).where(out["depth"] > 0.0, 0.0).where(out["depth"].notna())
        out["depth_volume"] = out["depth"] * out["underwater_volume"]
        out["volume_sum"] = out["underwater_volume"].rolling(params.debt_bars, min_periods=params.debt_bars).sum()
        out["depth_volume_sum"] = out["depth_volume"].rolling(params.debt_bars, min_periods=params.debt_bars).sum()
        out["debt_depth"] = out["depth_volume_sum"] / out["volume_sum"].where(out["volume_sum"] > 0.0)
        out["ceiling"] = out["debt_depth"].shift(1)
        out["deepest"] = out["depth"].rolling(params.debt_bars, min_periods=params.debt_bars).max().shift(1)
        out["breakout_high"] = data["high"].rolling(params.breakout_bars, min_periods=params.breakout_bars).max().shift(1)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["strength"] = ((out["deepest"] - out["depth"]) / out["deepest"].where(out["deepest"] > 0.0)).clip(0.0, 1.0)
        out["entry"] = (
            (out["depth"].shift(1) >= out["ceiling"].shift(1))
            & (out["depth"] < out["ceiling"])
            & (out["deepest"] >= params.min_depth)
            & (close > out["breakout_high"])
            & (out["atr"] > 0.0)
            & np.isfinite(out["strength"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, params.min_size, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        entry_atr = 0.0
        stop_price = np.nan
        exposure = params.min_size
        for i in range(n):
            if active:
                if i == entry_bar:
                    # The opening fill is known only on this bar, never on the decision bar.
                    stop_price = opens[i] * (1.0 + params.entry_slippage_bps / 10000.0) - params.stop_atr * entry_atr
                if not np.isfinite(stop_price) or not np.isfinite(closes[i]) or closes[i] < stop_price or i - entry_bar + 1 >= params.holding_bars:
                    active = False
                    continue
                signal[i] = 1
                size[i] = exposure
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(strength[i]):
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
                exposure = float(np.clip(strength[i], params.min_size, 1.0))
                signal[i] = 1
                size[i] = exposure
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0, 102.0, 105.0, 99.0, 98.0, 100.0],
        "close": [100.0, 110.0, 99.0, 97.0, 100.0, 100.0],
    }, index=index)
    derived = pd.DataFrame({
        "entry": [True, False, False, False, False, False],
        "atr": 2.0,
        "strength": 0.8,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    params = DrawdownParams(stop_atr=2.0)
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert np.allclose(result.loc[result["signal"] == 1, "size"], 0.8)
    params.holding_bars = 2
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
