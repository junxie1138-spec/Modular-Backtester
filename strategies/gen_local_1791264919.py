from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 60
    cost_window: int = 20
    volume_window: int = 20
    min_drawdown: float = 0.03
    volume_threshold: float = 1.25
    profit_target: float = 0.045
    max_hold: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791264919"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.peak_window + 1, params.cost_window + 2,
                   params.volume_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.peak_window, params.cost_window, params.volume_window,
               params.max_hold) < 1:
            raise ValueError("Windows and max_hold must be positive.")
        if not (0 < params.min_drawdown < 1 and params.volume_threshold > 0
                and 0 < params.profit_target < 1
                and 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid thresholds or position sizes.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        volume = data["volume"].clip(lower=0)
        out["previous_close"] = close.shift(1)
        out["change"] = close.diff()
        out["typical_price"] = (data["high"] + data["low"] + close) / 3.0
        out["distress_volume"] = volume.where(out["change"] < 0, 0.0)
        out.loc[out["change"].isna(), "distress_volume"] = np.nan
        out["distress_value"] = out["typical_price"] * out["distress_volume"]
        out["cost_volume"] = out["distress_volume"].rolling(params.cost_window).sum().shift(1)
        out["cost_value"] = out["distress_value"].rolling(params.cost_window).sum().shift(1)
        out["cost_basis"] = out["cost_value"] / out["cost_volume"].replace(0, np.nan)
        out["prior_peak"] = data["high"].rolling(params.peak_window).max().shift(1)
        out["drawdown"] = 1.0 - out["previous_close"] / out["prior_peak"].replace(0, np.nan)
        out["normal_volume"] = volume.rolling(params.volume_window).mean().shift(1)
        out["relative_volume"] = volume / out["normal_volume"].replace(0, np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).replace(0, np.nan)
        out["penetration"] = (close - out["cost_basis"]).abs() / out["true_range"]
        out["strength"] = (
            out["relative_volume"] / params.volume_threshold * out["penetration"]
        ).replace([np.inf, -np.inf], np.nan).clip(0, 1).fillna(0.0)
        out["entry_size"] = params.min_size + (
            params.max_size - params.min_size
        ) * out["strength"]
        ready = out[["cost_basis", "drawdown", "relative_volume", "true_range"]].notna().all(axis=1)
        eligible = ready & (out["drawdown"] >= params.min_drawdown) & (
            out["relative_volume"] >= params.volume_threshold
        )
        # Both closes straddle the same prior-session barrier.
        out["long_entry"] = (
            eligible & (out["previous_close"] <= out["cost_basis"])
            & (close > out["cost_basis"]) & (close > data["open"])
        ).fillna(False)
        out["short_entry"] = (
            eligible & (out["previous_close"] >= out["cost_basis"])
            & (close < out["cost_basis"]) & (close < data["open"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        long_entry = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        short_entry = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        entry_sizes = indicators["entry_size"].fillna(params.min_size).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        fill_bar = -1
        entry_price = np.nan
        held_size = params.min_size
        for i in range(n):
            if position:
                # Read the actual entry open only when that bar has arrived.
                if i == fill_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                if (not valid_entry or gain >= params.profit_target
                        or i - fill_bar + 1 >= params.max_hold):
                    position = 0
                    # Keep this close's exit; do not re-enter on the same close.
                else:
                    signals[i] = position
                    sizes[i] = held_size
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                fill_bar = i + 1
                entry_price = np.nan
                held_size = float(np.clip(entry_sizes[i], params.min_size, params.max_size))
                signals[i] = position
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10)
    close = np.array([100, 105, 100, 100, 95, 100, 100, 100, 100, 100], dtype=float)
    data = pd.DataFrame({
        "open": 100.0, "high": np.maximum(close, 100) + 1,
        "low": np.minimum(close, 100) - 1, "close": close, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "long_entry": [True, False, False, False, False, True, False, False, False, False],
        "short_entry": [False, False, False, True, False, False, False, False, False, False],
        "entry_size": [0.6, 0.6, 0.6, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4],
    }, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams(profit_target=0.04, max_hold=3)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0, 1, 1, 1, 0]
    assert np.allclose(result.loc[index[[1, 4, 6]], "size"], [0.6, 0.8, 0.4])
    assert result["size"].gt(0).all()
    assert not strategy.indicators(data, params)["long_entry"].any()
    assert not strategy.indicators(data, params)["short_entry"].any()


if __name__ == "__main__":
    _self_check()
