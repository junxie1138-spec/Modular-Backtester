from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalInnovationParams:
    smooth_bars: int = 5
    season_bars: int = 252
    min_phase_samples: int = 30
    entry_percentile: float = 0.80
    atr_bars: int = 14
    trail_atr: float = 1.5
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalInnovationParams]):
    strategy_id = "gen_local_1791306608"

    @classmethod
    def params_type(cls):
        return SeasonalInnovationParams

    def warmup_bars(self, params):
        return int(max(params.smooth_bars + 2, params.atr_bars) + params.season_bars + 1)

    def indicators(self, data, params):
        if not (params.smooth_bars >= 2 and params.atr_bars >= 2
                and params.season_bars >= params.min_phase_samples >= 2
                and 0 < params.entry_percentile < 1
                and params.trail_atr > 0 and 3 <= params.hold_days <= 5):
            raise ValueError("Invalid seasonal innovation parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["latent_level"] = out["log_close"].rolling(params.smooth_bars).mean()
        out["latent_slope"] = out["latent_level"].diff()
        out["slope_innovation"] = out["latent_slope"].diff()
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["innovation"] = out["slope_innovation"] / (
            out["atr"] / close).replace(0, np.nan)
        out["phase"] = np.minimum((data.index.day - 1) // 7, 3)
        out["phase_mean"] = np.nan
        out["phase_threshold"] = np.nan
        for phase in range(4):
            mask = out["phase"].eq(phase)
            history = out["innovation"].where(mask).shift(1)
            window = history.rolling(params.season_bars, min_periods=params.min_phase_samples)
            out.loc[mask, "phase_mean"] = window.mean().loc[mask]
            out.loc[mask, "phase_threshold"] = window.quantile(params.entry_percentile).loc[mask]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        entry = (
            indicators["phase_mean"].lt(0)
            & indicators["innovation"].gt(0)
            & indicators["innovation"].gt(indicators["phase_threshold"])
            & indicators["atr"].gt(0)
        ).fillna(False).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0
            if active:
                age += 1
                if valid:
                    # First in-trade close is the bar after the entry decision.
                    high_water = closes[i] if np.isnan(high_water) else max(high_water, closes[i])
                    candidate = high_water - params.trail_atr * atrs[i]
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                if not valid or closes[i] <= stop or age >= params.hold_days:
                    active = False
                else:
                    signal[i] = 1
                # Never reenter on an exit decision bar.
                continue
            if valid and entry[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = np.nan
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [99., 100., 104., 103., 102., 102., 102., 102.]}, index=index)
    ind = pd.DataFrame({
        "phase_mean": [-1.] * 8,
        "innovation": [2.] + [0.] * 7,
        "phase_threshold": [1.] * 8,
        "atr": [1., 1., 1., 10., 10., 10., 10., 10.],
    }, index=index)
    p = SeasonalInnovationParams(hold_days=5)
    result = strategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data["close"] = 100.
    p.hold_days = 3
    result = strategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
