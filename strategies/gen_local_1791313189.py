from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangepointParams:
    level_window: int = 5
    slope_window: int = 3
    impulse_window: int = 3
    percentile_window: int = 63
    entry_percentile: float = 0.80
    atr_window: int = 14
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ChangepointParams]):
    strategy_id = "gen_local_1791313189"

    @classmethod
    def params_type(cls) -> type[ChangepointParams]:
        return ChangepointParams

    def warmup_bars(self, params: ChangepointParams) -> int:
        return int(max(
            params.atr_window + 1,
            params.level_window + params.slope_window
            + params.impulse_window + params.percentile_window + 3,
        ))

    def indicators(self, data: pd.DataFrame, params: ChangepointParams) -> pd.DataFrame:
        windows = (params.level_window, params.slope_window,
                   params.impulse_window, params.percentile_window,
                   params.atr_window)
        if any(w < 1 for w in windows):
            raise ValueError("Indicator windows must be positive")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["latent_level"] = close.rolling(params.level_window).mean()
        out["level_change"] = out["latent_level"].diff()
        out["latent_slope"] = out["level_change"].rolling(params.slope_window).mean()
        out["previous_slope"] = out["latent_slope"].shift(1)
        out["innovation"] = out["latent_slope"].diff()
        out["innovation_sum"] = out["innovation"].rolling(params.impulse_window).sum()
        out["impulse"] = out["innovation_sum"] / out["atr"].where(out["atr"] > 0)
        out["impulse_magnitude"] = out["impulse"].abs()
        out["threshold"] = out["impulse_magnitude"].rolling(
            params.percentile_window
        ).quantile(params.entry_percentile).shift(1)
        out["minimum_innovation"] = out["innovation"].rolling(params.impulse_window).min()
        out["maximum_innovation"] = out["innovation"].rolling(params.impulse_window).max()
        out["long_entry"] = (
            (out["previous_slope"] <= 0)
            & (out["latent_slope"] > 0)
            & (out["minimum_innovation"] > 0)
            & (out["impulse"] > out["threshold"])
        )
        out["short_entry"] = (
            (out["previous_slope"] >= 0)
            & (out["latent_slope"] < 0)
            & (out["maximum_innovation"] < 0)
            & (-out["impulse"] > out["threshold"])
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ChangepointParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        long_entry = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        short_entry = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        extreme = np.nan
        stop = np.nan

        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = 0
                extreme = stop = np.nan
                continue

            if position:
                # Entry decisions fill next bar; only subsequent closes enter the trail.
                if position == 1:
                    extreme = close[i] if np.isnan(extreme) else max(extreme, close[i])
                    candidate = extreme - params.trail_atr * atr[i]
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    stopped = close[i] <= stop
                else:
                    # Shorts use the mirrored low-water mark and downward-ratcheting stop.
                    extreme = close[i] if np.isnan(extreme) else min(extreme, close[i])
                    candidate = extreme + params.trail_atr * atr[i]
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    stopped = close[i] >= stop
                if stopped or i - entry_bar >= params.hold_bars:
                    position = 0
                    extreme = stop = np.nan
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                entry_bar = i
                extreme = stop = np.nan

            signal[i] = position

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = ChangepointParams()
    index = pd.date_range("2020-01-01", periods=12)
    close = np.array([100, 103, 105, 102, 101, 100, 98, 97, 99, 100, 101, 102], dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close,
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "long_entry": False,
                        "short_entry": False}, index=index)
    ind.loc[index[0], "long_entry"] = True
    ind.loc[index[5], "short_entry"] = True
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0, 0]
    flat = data.copy()
    flat["close"] = 100.0
    ind["short_entry"] = False
    result = strategy.generate_signals(flat, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    actual = strategy.indicators(data, params)
    result = strategy.generate_signals(data, actual, None, params).data
    assert result["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
