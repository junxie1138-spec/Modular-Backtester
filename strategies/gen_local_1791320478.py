from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapInnovationParams:
    trend_window: int = 12
    atr_window: int = 14
    percentile_window: int = 126
    entry_percentile: float = 0.80
    stop_atr: float = 1.25
    hold_days: int = 2
    fill_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[GapInnovationParams]):
    strategy_id = "gen_local_1791320478"

    @classmethod
    def params_type(cls) -> type[GapInnovationParams]:
        return GapInnovationParams

    @staticmethod
    def warmup_bars(params: GapInnovationParams) -> int:
        return int(params.trend_window + params.atr_window + params.percentile_window + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapInnovationParams) -> pd.DataFrame:
        if min(params.trend_window, params.atr_window, params.percentile_window) < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one.")
        if params.stop_atr <= 0.0 or params.hold_days not in (1, 2):
            raise ValueError("Use a positive stop_atr and a one- or two-day holding period.")
        if not np.isfinite(params.fill_slippage_bps) or params.fill_slippage_bps < 0.0:
            raise ValueError("fill_slippage_bps must be finite and nonnegative.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        opening = data["open"].where(data["open"] > 0.0)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["risk_scale"] = (out["atr"].shift(1) / out["previous_close"]).where(lambda x: x > 0.0)
        out["overnight_return"] = np.log(opening / out["previous_close"])
        out["intraday_return"] = np.log(close / opening)
        # Mean increments estimate the slope of a price path stripped of overnight gaps.
        out["latent_slope"] = out["intraday_return"].rolling(params.trend_window, min_periods=params.trend_window).mean()
        out["prior_slope"] = out["latent_slope"].shift(1)
        out["slope_innovation"] = (out["intraday_return"] - out["prior_slope"]) / out["risk_scale"]
        out["gap_severity"] = -out["overnight_return"] / out["risk_scale"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["innovation_threshold"] = out["slope_innovation"].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_percentile)
        out["gap_threshold"] = out["gap_severity"].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_percentile)
        valid = out[["atr", "prior_slope", "slope_innovation", "gap_severity", "innovation_threshold", "gap_threshold"]].notna().all(axis=1)
        out["entry"] = (
            valid
            & (out["atr"] > 0.0)
            & (out["prior_slope"] > 0.0)
            & (out["intraday_return"] > 0.0)
            & (out["gap_severity"] > 0.0)
            & (out["gap_severity"] > out["gap_threshold"])
            & (out["slope_innovation"] > 0.0)
            & (out["slope_innovation"] > out["innovation_threshold"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapInnovationParams) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        fixed_stop = np.nan
        for i in range(len(data)):
            # A decision made yesterday enters at today's open.
            if not active and i > 0 and raw[i - 1] == 1:
                if not (np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(atr[i - 1])):
                    continue
                entry_price = opens[i] * (1.0 + params.fill_slippage_bps / 10000.0)
                fixed_stop = entry_price - params.stop_atr * atr[i - 1]
                active = True
                held = 0
            if active:
                held += 1
                # This level stays fixed for the entire trade.
                if not np.isfinite(closes[i]) or closes[i] < fixed_stop or held >= params.hold_days:
                    active = False
                    raw[i] = 0
                else:
                    raw[i] = 1
            elif entries[i]:
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({
        "open": [100.0] * 7,
        "high": [101.0] * 7,
        "low": [97.0] * 7,
        "close": [100.0, 98.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=index)
    ind = pd.DataFrame({"atr": [1.0] * 7, "entry": [True, False, False, True, False, False, False]}, index=index)
    params = GapInnovationParams(stop_atr=1.0, fill_slippage_bps=0.0)
    result = GeneratedStrategy.generate_signals(data, ind, None, params)
    assert result.data["signal"].tolist() == [0, 1, 0, 0, 1, 1, 0]
    assert (result.data["size"] > 0.0).all()
    params.hold_days = 1
    result = GeneratedStrategy.generate_signals(data, ind, None, params)
    assert result.data["signal"].tolist() == [0, 1, 0, 0, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
