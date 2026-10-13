from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    baseline_bars: int = 20
    recent_bars: int = 2
    change_threshold: float = 2.0
    volume_multiple: float = 1.5
    minimum_move: float = 0.002
    profit_target: float = 0.0075
    max_hold_bars: int = 2
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790978490"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if params.baseline_bars < 2 or params.recent_bars < 1:
            raise ValueError("Lookbacks must be positive; baseline must be at least two.")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be one or two.")
        values = np.array([
            params.change_threshold, params.volume_multiple,
            params.minimum_move, params.profit_target, params.size,
        ])
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Thresholds and size must be finite and positive.")
        return int(max(params.baseline_bars + params.recent_bars,
                       params.baseline_bars + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["bar_range"] = data["high"] - data["low"]
        out["body_pressure"] = (
            (data["close"] - data["open"])
            / out["bar_range"].where(out["bar_range"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        pressure = out["body_pressure"]
        n, k = params.baseline_bars, params.recent_bars
        out["recent_pressure"] = pressure.rolling(k, min_periods=k).mean()
        out["baseline_pressure"] = pressure.shift(k).rolling(n, min_periods=n).mean()
        out["baseline_std"] = pressure.shift(k).rolling(n, min_periods=n).std(ddof=0)
        out["change_score"] = (
            (out["recent_pressure"] - out["baseline_pressure"])
            / (out["baseline_std"].where(out["baseline_std"] > 1e-12)
               * np.sqrt(1.0 / k + 1.0 / n))
        ).replace([np.inf, -np.inf], np.nan)
        out["previous_close"] = data["close"].shift(1)
        out["daily_return"] = (
            data["close"]
            / out["previous_close"].where(out["previous_close"] > 0) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out["baseline_volume"] = data["volume"].shift(1).rolling(n, min_periods=n).median()
        out["volume_ratio"] = (
            data["volume"] / out["baseline_volume"].where(out["baseline_volume"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["change_direction"] = np.select([
            (out["change_score"] >= params.change_threshold)
            & (out["recent_pressure"] > 0) & (out["baseline_pressure"] <= 0),
            (out["change_score"] <= -params.change_threshold)
            & (out["recent_pressure"] < 0) & (out["baseline_pressure"] >= 0),
        ], [1, -1], default=0).astype(int)
        out["volume_direction"] = np.select([
            (out["volume_ratio"] >= params.volume_multiple)
            & (out["daily_return"] >= params.minimum_move),
            (out["volume_ratio"] >= params.volume_multiple)
            & (out["daily_return"] <= -params.minimum_move),
        ], [1, -1], default=0).astype(int)
        out["setup"] = out["change_direction"].where(
            out["change_direction"] == out["volume_direction"], 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        setup = indicators["setup"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if position != 0:
                # Yesterday's entry decision fills at today's open.
                if entry_bar < 0:
                    entry_bar, entry_price = i, opens[i]
                price_valid = np.isfinite(entry_price) and entry_price > 0
                gain = (
                    position * (closes[i] / entry_price - 1.0)
                    if price_valid and np.isfinite(closes[i]) else np.nan
                )
                # Close-based target decisions execute at the following open.
                if (not price_valid or gain >= params.profit_target
                        or i - entry_bar + 1 >= params.max_hold_bars):
                    position, entry_bar, entry_price = 0, -1, np.nan
                    # An exit always produces a flat decision, even during a setup.
                    continue
            elif setup[i] != 0:
                position = int(setup[i])
                entry_bar, entry_price = -1, np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), StrategyParams()
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 98.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    for side in (-1, 1):
        setup = pd.DataFrame({"setup": [side, 0, 0, 0, 0, 0]}, index=index)
        timed = strategy.generate_signals(data, setup, ctx, params).data
        assert timed["signal"].tolist() == [0, side, side, 0, 0, 0]
        profitable = data.copy()
        profitable.loc[index[1], "close"] = 100.0 * (
            1.0 + side * (params.profit_target + 0.001)
        )
        targeted = strategy.generate_signals(profitable, setup, ctx, params).data
        assert targeted["signal"].tolist() == [0, side, 0, 0, 0, 0]
        assert (targeted["size"] > 0).all()
    flat = strategy.indicators(data, params)
    assert (flat["setup"] == 0).all()


if __name__ == "__main__":
    _self_check()
