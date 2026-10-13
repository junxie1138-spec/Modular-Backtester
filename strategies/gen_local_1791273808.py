from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    depth_window: int = 63
    fit_window: int = 126
    forecast_bars: int = 18
    reversal_band: float = 0.006


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1791273808"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(max(200, params.depth_window + params.forecast_bars + params.fit_window))

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if params.depth_window < 2 or params.fit_window < 3 or params.forecast_bars < 1:
            raise ValueError("Lookbacks must be positive and regression needs at least three observations.")
        if not np.isfinite(params.reversal_band) or params.reversal_band <= 0:
            raise ValueError("reversal_band must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak"] = close.rolling(params.depth_window, min_periods=params.depth_window).max()
        out["depth"] = 1.0 - close / out["peak"]
        out["lagged_depth"] = out["depth"].shift(params.forecast_bars)
        # Each regression pair has its outcome observed by the current close.
        paired = out[["depth", "lagged_depth"]].where(out[["depth", "lagged_depth"]].notna().all(axis=1), axis=0)
        current = paired["depth"].rolling(params.fit_window, min_periods=params.fit_window)
        lagged = paired["lagged_depth"].rolling(params.fit_window, min_periods=params.fit_window)
        out["depth_mean"] = current.mean()
        out["lagged_mean"] = lagged.mean()
        out["lagged_variance"] = lagged.var()
        out["lagged_covariance"] = current.cov(paired["lagged_depth"])
        out["persistence"] = (out["lagged_covariance"] / out["lagged_variance"].where(out["lagged_variance"] > 1e-12)).clip(-0.98, 0.98)
        out["forecast_depth"] = (out["depth_mean"] + out["persistence"] * (out["depth"] - out["lagged_mean"])).clip(0.0, 1.0)
        out["forecast_recovery"] = out["depth"] - out["forecast_depth"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        score = indicators["forecast_recovery"].to_numpy(dtype=float)
        average = indicators["ma200"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        state = 0
        band = params.reversal_band
        for i in range(len(data)):
            if np.isfinite(score[i]):
                # Exit only on an opposite forecast; leave this decision bar flat.
                if state == 1 and score[i] <= -band:
                    state = 0
                elif state == -1 and score[i] >= band:
                    state = 0
                elif state == 0 and np.isfinite(average[i]) and np.isfinite(close[i]):
                    if score[i] >= band and close[i] > average[i]:
                        state = 1
                    elif score[i] <= -band and close[i] < average[i]:
                        state = -1
            signal[i] = state
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [210.0, 210.0, 190.0, 190.0, 210.0, 210.0, 210.0]}, index=index)
    derived = pd.DataFrame({"ma200": 200.0, "forecast_recovery": [0.01, 0.0, -0.01, -0.01, 0.01, np.nan, 0.01]}, index=index)
    result = GeneratedStrategy().generate_signals(data, derived, None, DrawdownParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
