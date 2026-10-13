from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    shock_bars: int = 126
    shock_percentile: float = 0.20
    season_bars: int = 252
    min_samples: int = 4
    season_percentile: float = 0.60
    atr_bars: int = 14
    stop_atr: float = 1.0
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790917252"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.shock_bars + params.season_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (min(params.shock_bars, params.season_bars, params.atr_bars, params.min_samples) < 1
                or not 0 < params.shock_percentile < 1
                or not 0 < params.season_percentile < 1
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0
                or params.hold_bars not in (1, 2)):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["shock_cutoff"] = out["return"].rolling(params.shock_bars).quantile(params.shock_percentile).shift(1)
        out["shock"] = (out["return"] < 0) & (out["return"] <= out["shock_cutoff"])
        out["recovery_return"] = out["return"].where(out["shock"].shift(1, fill_value=False))
        weekday = data.index.dayofweek
        columns = []
        for day in range(5):
            name = f"weekday_recovery_{day}"
            columns.append(name)
            out[name] = out["recovery_return"].where(weekday == day).rolling(
                params.season_bars, min_periods=params.min_samples
            ).mean().shift(1)
        out["season_cutoff"] = out[columns].quantile(params.season_percentile, axis=1)
        # ponytail: weekday approximation skips exchange holidays; add a calendar if holiday accuracy matters.
        next_day = (data.index + pd.offsets.BDay(1)).dayofweek
        out["next_recovery"] = np.nan
        for day, name in enumerate(columns):
            out.loc[next_day == day, "next_recovery"] = out.loc[next_day == day, name]
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        ready = np.arange(len(data)) >= self.warmup_bars(params)
        out["entry"] = (
            ready & out["shock"] & out[columns].notna().all(axis=1)
            & (out["next_recovery"] > 0)
            & (out["next_recovery"] >= out["season_cutoff"])
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        desired = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if entry_bar < 0 and i > 0 and desired[i - 1] == 1:
                if not (np.isfinite(opens[i]) and opens[i] > 0
                        and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                entry_bar = i
                stop = opens[i] - params.stop_atr * atr[i - 1]
            if entry_bar >= 0:
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    entry_bar = -1
                    continue
                desired[i] = 1
            elif entries[i]:
                desired[i] = 1
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.bdate_range("2020-01-01", periods=6)
    sample = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 97.0,
        "close": [100.0, 98.0, 100.0, 100.0, 100.0, 100.0], "volume": 1000.0,
    }, index=index)
    derived = pd.DataFrame({"entry": True, "atr": 1.0}, index=index)
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = GeneratedStrategy().generate_signals(sample, derived, context, Params()).data
    assert result["signal"].tolist() == [0, 1, 0, 1, 1, 0]
    assert result["size"].gt(0).all()
