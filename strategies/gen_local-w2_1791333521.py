from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAccelerationParams:
    impulse_window: int = 4
    vol_window: int = 20
    reversal_threshold: float = 0.35
    target_vol: float = 0.12
    max_size: float = 0.95
    min_confidence: float = 0.10


class GeneratedStrategy(BaseStrategy[GapAccelerationParams]):
    strategy_id = "gen_local-w2_1791333521"

    @classmethod
    def params_type(cls):
        return GapAccelerationParams

    def warmup_bars(self, params):
        if params.impulse_window < 2 or params.vol_window < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        values = (params.reversal_threshold, params.target_vol, params.max_size, params.min_confidence)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Threshold and sizing parameters must be finite and positive.")
        if params.max_size > 1 or params.min_confidence > 1:
            raise ValueError("Size and confidence caps cannot exceed one.")
        return int(max(2 * params.impulse_window + 1, params.vol_window + 1))

    def indicators(self, data, params):
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_close"] = out["prior_close"].where(out["prior_close"] > 0)
        out["gap"] = (data["open"] / out["prior_close"] - 1).replace([np.inf, -np.inf], np.nan)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["annual_vol"] = out["return"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["gap_rms"] = np.sqrt(out["gap"].pow(2).rolling(params.vol_window, min_periods=params.vol_window).mean())
        out["recent_gap_sum"] = out["gap"].rolling(params.impulse_window, min_periods=params.impulse_window).sum()
        out["previous_gap_sum"] = out["recent_gap_sum"].shift(params.impulse_window)
        out["gap_scale"] = out["gap_rms"].clip(lower=1e-6) * np.sqrt(2.0 * params.impulse_window)
        out["pressure"] = (out["recent_gap_sum"] - out["previous_gap_sum"]) / out["gap_scale"]
        out["ready"] = out[["pressure", "annual_vol"]].notna().all(axis=1)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        pressure = indicators["pressure"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ready = indicators["ready"].fillna(False).astype(bool)
        candidate = np.where(ready & pressure.ge(params.reversal_threshold), 1,
                             np.where(ready & pressure.le(-params.reversal_threshold), -1, 0))
        desired = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(candidate):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            desired[i] = position

        df = pd.DataFrame({"signal": desired}, index=data.index)
        confidence = np.tanh((df["signal"] * pressure).clip(lower=0.0)).clip(lower=params.min_confidence, upper=1.0)
        volatility = indicators["annual_vol"].replace([np.inf, -np.inf], np.nan).fillna(0.02).clip(lower=0.02)
        size = (params.target_vol / volatility * confidence).clip(lower=1e-6, upper=params.max_size)
        df["size"] = size.shift(1).fillna(params.max_size).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapAccelerationParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "pressure": [1.0, 0.0, 0.2, -1.0, -1.0, -0.1, 1.0, 1.0, 0.0],
        "annual_vol": [0.2] * 9,
        "ready": [True] * 9,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0, 1]
    assert result["size"].gt(0).all() and result["size"].le(params.max_size).all()
    assert np.isclose(result["size"].iloc[1], params.target_vol / 0.2 * np.tanh(1.0))


if __name__ == "__main__":
    _self_check()
