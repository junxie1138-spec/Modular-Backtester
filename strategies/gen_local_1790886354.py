from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    depth_atr: float = 3.0
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1790886354"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return 82

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if not all(np.isfinite(v) and v > 0 for v in (params.depth_atr, params.trail_atr)):
            raise ValueError("Parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean().replace(0, np.nan)
        out["peak"] = data["high"].rolling(63, min_periods=63).max()
        out["depth"] = (out["peak"] - data["close"]) / out["atr"]
        out["depth_change"] = out["depth"].diff()
        out["underwater"] = (out["depth"] > 1.0).astype(float).where(out["depth"].notna())
        out["underwater_share"] = out["underwater"].rolling(20, min_periods=20).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: DrawdownParams
    ) -> SignalFrame:
        depth = indicators["depth"]
        change = indicators["depth_change"]
        share = indicators["underwater_share"]
        ready = np.isfinite(indicators[["atr", "depth", "depth_change", "underwater_share"]]).all(axis=1)
        long_entry = ready & (depth >= params.depth_atr) & (change < 0) & (share <= 0.55)
        short_entry = ready & (depth >= params.depth_atr) & (change > 0) & (share >= 0.80)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        direction, age = 0, 0
        entry, stop_gain, best_gain = np.nan, 0.0, 0.0
        armed = False
        for i in range(len(data)):
            if not ready.iloc[i] or not np.isfinite(closes[i]) or closes[i] <= 0:
                direction, entry = 0, np.nan
                continue
            if direction:
                # Entry follows the previous close's decision at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        direction, entry = 0, np.nan
                        continue
                    stop_gain = -2.5 * atrs[i - 1]
                    best_gain, age, armed = 0.0, 0, False
                age += 1
                gain = direction * (closes[i] - entry)
                best_gain = max(best_gain, gain)
                armed = armed or best_gain >= 0.01 * entry
                if armed:
                    # Signed profit coordinates: stop rises for BOTH directions.
                    stop_gain = max(stop_gain, 0.0, best_gain - params.trail_atr * atrs[i])
                # Close-confirmed exits execute at the following open.
                if gain <= stop_gain or age >= 10:
                    direction, entry = 0, np.nan
            elif candidates[i]:
                direction = int(candidates[i])
                entry = np.nan
            signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    prices = pd.Series([100., 101., 103., 101., 100., 100., 98., 100., 100., 100.],
                       index=pd.date_range("2020-01-01", periods=10))
    bars = pd.DataFrame({"open": prices, "high": prices + 0.5, "low": prices - 0.5,
                         "close": prices, "volume": 10000.})
    derived = pd.DataFrame({"atr": 1., "depth": 4., "depth_change": 0.,
                            "underwater_share": 0.6}, index=bars.index)
    derived.loc[bars.index[0], ["depth_change", "underwater_share"]] = [-1., 0.4]
    derived.loc[bars.index[4], ["depth_change", "underwater_share"]] = [1., 0.9]
    result = GeneratedStrategy().generate_signals(bars, derived, None, DrawdownParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0, 0]
