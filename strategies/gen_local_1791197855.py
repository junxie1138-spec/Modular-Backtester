from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    rank_window: int = 63
    analog_window: int = 252
    neighbors: int = 9
    max_distance: float = 0.20
    min_edge: float = 0.004
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791197855"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        if (
            params.rank_window < 2
            or params.analog_window < params.neighbors * params.hold_bars
            or params.neighbors < 1
            or params.hold_bars not in (3, 4, 5)
            or not 0 < params.max_distance <= 1
            or not 0 < params.min_edge < 1
            or not 0 < params.profit_target < 1
        ):
            raise ValueError("Invalid analog parameters")
        return int(max(200, params.rank_window + 3 + params.hold_bars + params.analog_window))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        warmup = self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_rank"] = out["return"].rolling(params.rank_window).rank(pct=True)
        out["rank_lag1"] = out["return_rank"].shift(1)
        out["rank_lag2"] = out["return_rank"].shift(2)
        out["ma200"] = close.rolling(200).mean()
        out["regime"] = np.select(
            [close > out["ma200"], close < out["ma200"]], [1, -1], default=0
        )
        # At t, this outcome belongs to the shape ending at t - hold_bars.
        out["mature_return"] = (
            close / opening.shift(params.hold_bars - 1) - 1
        ).replace([np.inf, -np.inf], np.nan)
        shapes = out[["rank_lag2", "rank_lag1", "return_rank"]].to_numpy()
        regimes = out["regime"].to_numpy()
        outcomes = out["mature_return"].to_numpy()
        forecast = np.full(len(data), np.nan)
        # ponytail: bounded O(bars * analog_window) scan; index shapes if history grows.
        for i in range(warmup - 1, len(data)):
            if regimes[i] == 0 or not np.isfinite(shapes[i]).all():
                continue
            last = i - params.hold_bars
            first = max(0, last - params.analog_window + 1)
            # Nonoverlapping outcome intervals reduce duplicate analog evidence.
            candidates = np.arange(last, first - 1, -params.hold_bars)
            labels = outcomes[candidates + params.hold_bars]
            distance = np.sqrt(np.mean((shapes[candidates] - shapes[i]) ** 2, axis=1))
            valid = (
                np.isfinite(distance)
                & np.isfinite(labels)
                & (regimes[candidates] == regimes[i])
                & (distance <= params.max_distance)
            )
            selected = np.flatnonzero(valid)
            if len(selected) >= params.neighbors:
                nearest = selected[np.argsort(distance[selected], kind="stable")[:params.neighbors]]
                forecast[i] = labels[nearest].mean()
        out["forecast"] = forecast
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(warmup - 1, len(data)):
            if side:
                # Read the entry open only when that bar has actually arrived.
                if i == entry_bar:
                    entry_price = opening[i]
                expired = i - entry_bar + 1 >= params.hold_bars
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = side * (close[i] / entry_price - 1) if valid_entry else np.nan
                if not valid_entry or expired or gain >= params.profit_target:
                    side = 0
                    continue
                signal[i] = side
                continue
            if not (np.isfinite(close[i]) and close[i] > 0 and np.isfinite(ma[i]) and np.isfinite(forecast[i])):
                continue
            if close[i] > ma[i] and forecast[i] >= params.min_edge:
                side = 1
            elif close[i] < ma[i] and forecast[i] <= -params.min_edge:
                side = -1
            if side:
                entry_bar = i + 1
                entry_price = np.nan
                signal[i] = side
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = AnalogParams()
    n = 420
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(0.0001 * np.arange(n) + 0.025 * np.sin(np.arange(n) / 7))
    data = pd.DataFrame(
        {"open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 1000.0},
        index=index,
    )
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:400], params)
    assert np.allclose(full.iloc[:400].to_numpy(), prefix.to_numpy(), equal_nan=True)
    assert full["forecast"].notna().any()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    live = strategy.generate_signals(data, full, ctx, params).data
    partial = strategy.generate_signals(data.iloc[:400], prefix, ctx, params).data
    assert live.iloc[:400].equals(partial)
    data[["open", "close"]] = 100.0
    data["high"] = 103.0
    data["low"] = 97.0
    start = strategy.warmup_bars(params) - 1
    data.iloc[start + 1, data.columns.get_loc("close")] = 102.0
    data.iloc[start + 4, data.columns.get_loc("close")] = 98.0
    probe = pd.DataFrame({"forecast": 0.0, "ma200": 90.0}, index=index)
    probe.iloc[[start, start + 6], 0] = params.min_edge * 2
    probe.iloc[start + 3] = [-params.min_edge * 2, 110.0]
    result = strategy.generate_signals(data, probe, ctx, params).data
    assert result["signal"].iloc[start:start + 12].tolist() == [0, 1, 0, 0, -1, 0, 0, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    assert set(result["signal"].unique()) <= {-1, 0, 1}
