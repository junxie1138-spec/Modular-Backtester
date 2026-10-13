from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 63
    percentile_window: int = 126
    stress_percentile: float = 0.8
    breadth_horizons: int = 5
    breadth_window: int = 5
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791343222"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.peak_window + params.percentile_window,
            params.breadth_horizons + params.breadth_window + 2,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (
            min(params.peak_window, params.percentile_window,
                params.breadth_horizons, params.breadth_window, params.max_hold) < 1
            or not 0 < params.stress_percentile < 1
            or not np.isfinite(params.profit_target)
            or params.profit_target <= 0
        ):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out = pd.DataFrame(index=data.index)
        out["peak"] = close.rolling(params.peak_window).max()
        out["depth"] = 1.0 - close / out["peak"]
        out["depth_change"] = out["depth"].diff()
        out["depth_threshold"] = out["depth"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.stress_percentile)
        horizon_columns = []
        for lag in range(1, params.breadth_horizons + 1):
            name = f"underwater_{lag}"
            previous = close.shift(lag)
            out[name] = (close < previous).astype(float).where(
                close.notna() & previous.notna()
            )
            horizon_columns.append(name)
        out["breadth"] = out[horizon_columns].sum(
            axis=1, min_count=params.breadth_horizons
        ) / params.breadth_horizons
        out["smoothed_breadth"] = out["breadth"].rolling(params.breadth_window).mean()
        out["breadth_change"] = out["smoothed_breadth"].diff()
        out["previous_breadth_change"] = out["breadth_change"].shift(1)
        stressed = (out["depth"] > 0) & (out["depth"] >= out["depth_threshold"])
        out["long_entry"] = (
            stressed & (out["depth_change"] < 0)
            & (out["breadth_change"] < 0) & (out["previous_breadth_change"] >= 0)
        ).fillna(False)
        out["short_entry"] = (
            stressed & (out["depth_change"] > 0)
            & (out["breadth_change"] > 0) & (out["previous_breadth_change"] <= 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if entry_bar == -1:
                    # The prior close's decision fills at this bar's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_close else 0.0
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    position = 0
                    entry_bar = -1
                    # Exit decisions take effect at the next open.
                    continue
            elif np.isfinite(closes[i]) and closes[i] > 0:
                position = 1 if longs[i] else (-1 if shorts[i] else 0)
                entry_bar = -1
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Check next-open entry, long profit-taking, and a two-session short expiry.
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0] * 7, "high": [104.0] * 7, "low": [99.0] * 7,
        "close": [100.0, 103.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=index)
    entries = pd.DataFrame({
        "long_entry": [True, False, False, False, False, False, False],
        "short_entry": [False, False, False, True, False, False, False],
    }, index=index)
    result = strategy.generate_signals(
        data, entries, None, Params(profit_target=0.02, max_hold=2)
    ).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
