from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityScalingParams:
    vol_window: int = 40
    aggregate_bars: int = 5
    anchor_bars: int = 20
    regime_gap: float = 0.15
    entry_threshold: float = 0.75
    profit_target: float = 0.025
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VolatilityScalingParams]):
    strategy_id = "gen_local-w2_1791263797"

    @classmethod
    def params_type(cls):
        return VolatilityScalingParams

    def warmup_bars(self, params):
        return int(max(params.anchor_bars, params.vol_window + params.aggregate_bars))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["log_price"] = np.log(data["close"].where(data["close"] > 0))
        out["log_return"] = out["log_price"].diff()
        out["daily_vol"] = out["log_return"].rolling(params.vol_window).std(ddof=0)
        out["aggregate_return"] = out["log_return"].rolling(params.aggregate_bars).sum()
        out["aggregate_vol"] = out["aggregate_return"].rolling(params.vol_window).std(ddof=0)
        daily_vol = out["daily_vol"].where(out["daily_vol"] > 0)
        out["vol_ratio"] = out["aggregate_vol"] / (daily_vol * np.sqrt(params.aggregate_bars))
        out["center"] = out["log_price"].rolling(params.anchor_bars).mean()
        out["tension"] = (out["log_price"] - out["center"]) / (daily_vol * np.sqrt(params.anchor_bars))
        out["momentum"] = out["aggregate_return"] / (daily_vol * np.sqrt(params.aggregate_bars))
        out["score"] = np.where(
            out["vol_ratio"] > 1.0 + params.regime_gap,
            out["momentum"],
            np.where(out["vol_ratio"] < 1.0 - params.regime_gap, -out["tension"], 0.0),
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        out["score"] = out["score"].where(out["vol_ratio"].notna(), 0.0).fillna(0.0)
        out["entry"] = np.where(
            out["score"].abs() >= params.entry_threshold, np.sign(out["score"]), 0
        ).astype(int)
        out["entry_size"] = (out["score"].abs() / 2.0).clip(0.2, 0.95)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.2, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].fillna(0.2).to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        trade_size = 0.2

        for i in range(n):
            exited = False
            if position and i == entry_bar:
                # Record the actual next-bar open only when that bar arrives.
                entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                    exited = True
            if position and i >= entry_bar:
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
                    exited = True
            if not position and not exited and entries[i]:
                position = int(entries[i])
                entry_bar = i + 1
                trade_size = float(np.clip(entry_sizes[i], 0.2, 0.95))
            signals[i] = position
            sizes[i] = trade_size if position else 0.2

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = VolatilityScalingParams()
    idx = pd.date_range("2020-01-01", periods=12)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (-1, 1):
        data = pd.DataFrame(100.0, index=idx, columns=["open", "high", "low", "close", "volume"])
        ind = pd.DataFrame({"entry": [direction] + [0] * 11, "entry_size": 0.4}, index=idx)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0] + [direction] * 8 + [0] * 3
        assert (result.loc[result["signal"] != 0, "size"] == 0.4).all()
        data.loc[idx[2], "close"] = 100.0 * (1.0 + direction * 0.03)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction] + [0] * 9
    data.loc[:, "close"] = 100.0
    data.loc[idx[1], "open"] = 102.0
    data.loc[idx[2], "close"] = 103.0
    data.loc[idx[3], "close"] = 106.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1] + [0] * 8
    flat = pd.DataFrame(100.0, index=pd.date_range("2020-01-01", periods=100),
                        columns=["open", "high", "low", "close", "volume"])
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data
    assert (result["signal"] == 0).all() and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
