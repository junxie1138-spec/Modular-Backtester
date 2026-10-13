from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    peak_window: int = 30
    pressure_window: int = 5
    percentile_window: int = 126
    trend_window: int = 60
    trend_lag: int = 10
    atr_window: int = 14
    depth_quantile: float = 0.70
    transmission_quantile: float = 0.35
    initial_stop_atr: float = 2.0
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local_1790907648"

    @classmethod
    def params_type(cls) -> type[AbsorptionParams]:
        return AbsorptionParams

    @staticmethod
    def warmup_bars(params: AbsorptionParams) -> int:
        return int(max(
            params.peak_window + params.pressure_window + params.percentile_window + 1,
            params.trend_window + params.trend_lag,
            params.atr_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["return"] = close.div(out["previous_close"].replace(0.0, np.nan)) - 1.0
        out["peak"] = close.rolling(p.peak_window, min_periods=p.peak_window).max()
        out["depth"] = (1.0 - close.div(out["peak"].replace(0.0, np.nan))).clip(lower=0.0)
        out["downside_return"] = (-out["return"]).clip(lower=0.0)
        out["selling_pressure"] = out["downside_return"].rolling(
            p.pressure_window, min_periods=p.pressure_window
        ).sum()
        out["depth_expansion"] = out["depth"].diff(p.pressure_window).clip(lower=0.0)
        out["transmission"] = out["depth_expansion"].div(
            out["selling_pressure"].replace(0.0, np.nan)
        )
        out["depth_threshold"] = out["depth"].rolling(
            p.percentile_window, min_periods=p.percentile_window
        ).quantile(p.depth_quantile).shift(1)
        out["transmission_threshold"] = out["transmission"].rolling(
            p.percentile_window, min_periods=p.percentile_window
        ).quantile(p.transmission_quantile).shift(1)
        out["trend_average"] = close.rolling(p.trend_window, min_periods=p.trend_window).mean()
        out["previous_trend_average"] = out["trend_average"].shift(p.trend_lag)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(p.atr_window, min_periods=p.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[[
            "depth", "depth_threshold", "transmission", "transmission_threshold",
            "trend_average", "previous_trend_average", "atr",
        ]].notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["depth"] > 0.0)
            & (out["depth"] >= out["depth_threshold"])
            & (out["transmission"] <= out["transmission_threshold"])
            & (close > out["trend_average"])
            & (out["trend_average"] > out["previous_trend_average"])
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AbsorptionParams,
    ) -> SignalFrame:
        p = params
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        holding = False
        pending = False
        armed = False
        entry_price = 0.0
        peak_close = 0.0
        stop = 0.0
        entry_atr = 0.0
        age = 0

        for i in range(len(data)):
            # A previous close's entry decision fills at this bar's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    entry_price = opens[i]
                    peak_close = entry_price
                    stop = entry_price - p.initial_stop_atr * entry_atr
                    holding = True
                    armed = False
                    age = 0

            exited = False
            if holding:
                age += 1
                valid_close = np.isfinite(closes[i]) and closes[i] > 0.0
                # Stops are close-observed; exit orders fill on the next bar.
                if age >= p.max_hold_bars or (valid_close and closes[i] <= stop):
                    holding = False
                    exited = True
                elif valid_close:
                    peak_close = max(peak_close, closes[i])
                    armed = armed or closes[i] >= entry_price * (1.0 + p.breakeven_pct)
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atrs[i]) and atrs[i] > 0.0:
                            stop = max(stop, peak_close - p.trail_atr * atrs[i])

            if not holding and not exited and entries[i]:
                if np.isfinite(atrs[i]) and atrs[i] > 0.0:
                    pending = True
                    entry_atr = atrs[i]

            signals[i] = int(holding or pending)

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [101.0, 103.0, 105.0, 101.0, 101.0, 101.0],
        "low": [99.0] * 6,
        "close": [100.0, 102.0, 104.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 6,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, AbsorptionParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0.0).all()
    warmup = GeneratedStrategy.indicators(data, AbsorptionParams())
    assert not warmup["entry"].any()


if __name__ == "__main__":
    _self_check()
