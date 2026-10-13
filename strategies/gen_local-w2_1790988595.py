from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 16
    percentile_window: int = 126
    center_window: int = 20
    atr_window: int = 14
    expansion_percentile: float = 0.80
    stretch_percentile: float = 0.85
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790988595"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.pattern_window + params.percentile_window + 8,
            params.center_window + params.percentile_window + 1,
            params.atr_window + params.percentile_window + 2,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None)
        for lag in range(4):
            out[f"return_lag_{lag}"] = out["return"].shift(lag)
        a, b, c, d = (out[f"return_lag_{lag}"] for lag in (3, 2, 1, 0))
        valid = np.isfinite(out[[f"return_lag_{lag}" for lag in range(4)]]).all(axis=1)
        valid &= (a != b) & (a != c) & (a != d) & (b != c) & (b != d) & (c != d)
        # Lehmer code identifies all 24 strict orderings of four returns.
        out["ordinal_code"] = (
            (a > b).astype(int) * 6 + (a > c).astype(int) * 6
            + (a > d).astype(int) * 6 + (b > c).astype(int) * 2
            + (b > d).astype(int) * 2 + (c > d).astype(int)
        ).where(valid)
        seen_columns = []
        for state in range(24):
            name = f"seen_{state}"
            out[name] = (
                out["ordinal_code"].eq(state).astype(float).where(valid)
                .rolling(params.pattern_window, min_periods=params.pattern_window).max()
            )
            seen_columns.append(name)
        out["complexity"] = out[seen_columns].sum(axis=1, min_count=24) / 24.0
        out["expansion"] = out["complexity"].diff(3)
        out["expansion_threshold"] = (
            out["expansion"].shift(1)
            .rolling(params.percentile_window, min_periods=params.percentile_window)
            .quantile(params.expansion_percentile)
        )
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(
            data[["high", "low", "close"]].notna().all(axis=1)
            & out["previous_close"].notna()
        )
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().replace(0.0, np.nan)
        out["center"] = close.rolling(
            params.center_window, min_periods=params.center_window
        ).median()
        out["stretch"] = (close - out["center"]) / out["atr"]
        out["absolute_stretch"] = out["stretch"].abs()
        out["stretch_threshold"] = (
            out["absolute_stretch"].shift(1)
            .rolling(params.percentile_window, min_periods=params.percentile_window)
            .quantile(params.stretch_percentile)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        required = ["expansion", "expansion_threshold", "stretch",
                    "stretch_threshold", "return", "atr"]
        ready = indicators[required].notna().all(axis=1)
        release = (
            ready
            & (indicators["expansion"] > indicators["expansion_threshold"])
            & (indicators["expansion"] > 0.0)
            & (indicators["stretch"].abs() > indicators["stretch_threshold"])
        )
        # Require a first closing move toward equilibrium.
        long_entry = release & (indicators["stretch"] < 0.0) & (indicators["return"] > 0.0)
        short_entry = release & (indicators["stretch"] > 0.0) & (indicators["return"] < 0.0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        pending_entry = False
        entry_atr, stop = np.nan, np.nan
        for i in range(len(data)):
            if side != 0:
                if pending_entry:
                    # Today's open is the fill reference for yesterday's decision.
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        side = 0
                        pending_entry = False
                        continue
                    entry_price = opens[i]
                    stop = entry_price - side * params.stop_atr * entry_atr
                    pending_entry = False
                age += 1
                stopped = np.isfinite(closes[i]) and side * (closes[i] - stop) < 0.0
                if stopped or age >= params.hold_bars or not np.isfinite(closes[i]):
                    side = 0
                    # Stay flat on the exit decision; no same-close reentry.
                    continue
                raw[i] = side
            elif candidates[i] != 0 and np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                side = int(candidates[i])
                entry_atr = atrs[i]
                age = 0
                pending_entry = True
                raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
