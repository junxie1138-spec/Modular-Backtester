from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 12
    profit_target: float = 0.008


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790856779"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        n = params.lookback
        if n < 3 or not np.isfinite(params.profit_target) or not 0 < params.profit_target < 1:
            raise ValueError("lookback must be >= 3 and profit_target must be between 0 and 1")
        out = pd.DataFrame(index=data.index)
        out["midpoint"] = ((data["high"] + data["low"]) / 2).replace([np.inf, -np.inf], np.nan)
        out["lower"] = data["low"].rolling(n, min_periods=n).min()
        out["upper"] = data["high"].rolling(n, min_periods=n).max()
        out["width"] = (out["upper"] - out["lower"]).where(out["upper"] > out["lower"])
        out["relative_position"] = (data["close"] - out["lower"]) / out["width"]

        # Causal rolling OLS slope of the trading-range midpoint.
        x = np.arange(n, dtype=float) - (n - 1) / 2
        weights = x / np.dot(x, x)
        values = out["midpoint"].to_numpy(dtype=float)
        slopes = np.convolve(values, weights[::-1], mode="full")[:len(data)] if len(data) else np.empty(0)
        out["latent_slope"] = slopes
        out.iloc[:n - 1, out.columns.get_loc("latent_slope")] = np.nan
        out["slope_innovation"] = out["latent_slope"].diff()
        out["scaled_innovation"] = n * out["slope_innovation"] / out["width"]
        out["previous_innovation"] = out["scaled_innovation"].shift(1)
        long_entry = (
            (out["relative_position"] <= 0.35)
            & (out["latent_slope"] < 0)
            & (out["scaled_innovation"] > 0.02)
            & (out["previous_innovation"] <= 0.02)
        )
        short_entry = (
            (out["relative_position"] >= 0.65)
            & (out["latent_slope"] > 0)
            & (out["scaled_innovation"] < -0.02)
            & (out["previous_innovation"] >= -0.02)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1)
        out["entry"] = np.select([long_entry & valid, short_entry & valid], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if side:
                # Read the execution bar's open only after that bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0
                # Gross directional price gain relative to the entry open.
                gain = side * (closes[i] / entry_price - 1) if valid_price else np.nan
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                # Count the entry session as session one; exit next open.
                timed_out = i - entry_bar + 1 >= 2
                if not valid_price or target_hit or timed_out:
                    side = 0
                    entry_price = np.nan
                raw[i] = side
                # An exit always leaves a flat decision bar before re-entry.
                continue
            if entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
                raw[i] = side

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0,
         "close": [100.0, 101.0, 100.0, 100.0, 99.0, 100.0], "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=14)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 0, 0, -1, 0]
    data["close"] = 100.0
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert (signals["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
