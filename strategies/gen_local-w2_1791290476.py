from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    fast_window: int = 5
    slow_window: int = 40
    atr_window: int = 14
    history_bars: int = 180
    bandwidth: float = 0.22
    forecast_threshold: float = 0.15
    min_analogs: int = 6
    breakeven_pct: float = 0.006
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791290476"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if not (2 <= params.fast_window <= params.slow_window
                and params.atr_window >= 1 and params.history_bars >= 8
                and params.min_analogs >= 1):
            raise ValueError("Invalid lookbacks or analog count")
        values = (params.bandwidth, params.forecast_threshold,
                  params.breakeven_pct, params.trail_atr)
        if not np.isfinite(values).all() or not (
                params.bandwidth > 0 and params.forecast_threshold >= 0
                and params.breakeven_pct > 0 and params.trail_atr > 0):
            raise ValueError("Invalid forecast or stop parameters")
        return int(max(params.slow_window, params.atr_window)
                   + params.history_bars + 5)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["fast_vol"] = out["return"].rolling(params.fast_window).std()
        out["slow_vol"] = out["return"].rolling(params.slow_window).std()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        ratio = out["fast_vol"] / out["slow_vol"].replace(0.0, np.nan)
        out["shape_now"] = np.log(ratio.where(ratio > 0))
        out["shape_middle"] = out["shape_now"].shift(2)
        out["shape_start"] = out["shape_now"].shift(4)
        shape = out[["shape_now", "shape_middle", "shape_start"]]
        total = pd.Series(0.0, index=data.index)
        weight_sum = total.copy()
        count = total.copy()
        # Three-bar spacing avoids sharing two-day outcome intervals.
        for lag in range(8, params.history_bars + 1, 3):
            distance = (shape - shape.shift(lag)).pow(2).sum(axis=1, min_count=3) / 3.0
            outcome = (close.shift(lag - 2) - close.shift(lag)) / out["atr"].shift(lag).replace(0.0, np.nan)
            valid = distance.le(params.bandwidth ** 2) & np.isfinite(outcome)
            weight = np.exp(-distance / (2.0 * params.bandwidth ** 2)).where(valid, 0.0)
            total += weight * outcome.where(valid, 0.0)
            weight_sum += weight
            count += valid.astype(int)
        out["analog_count"] = count
        out["forecast_atr"] = total / weight_sum.replace(0.0, np.nan)
        out["evidence"] = (
            out["analog_count"].ge(params.min_analogs)
            & out["forecast_atr"].gt(params.forecast_threshold)
            & out["atr"].gt(0)
        ).fillna(False)
        out.iloc[:self.warmup_bars(params), out.columns.get_loc("evidence")] = False
        out["confirmed"] = out["evidence"] & out["evidence"].shift(1, fill_value=False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        raw = np.zeros(len(data), dtype=int)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        active = False
        entry = np.nan
        stop = np.nan
        peak = np.nan
        armed = False
        age = 0
        for i, (opening, high, low, close) in enumerate(prices):
            if active:
                # Entry decision was made yesterday; anchor to today's open.
                if age == 0:
                    entry = opening
                    peak = entry
                    stop = entry - params.trail_atr * atr[i - 1]
                age += 1
                valid = np.isfinite([opening, high, low, close, atr[i]]).all() and atr[i] > 0
                # Test the previously established stop before updating it.
                stopped = not valid or opening <= stop or low <= stop
                if valid and not stopped:
                    peak = max(peak, high)
                    armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                        stopped = close <= stop
                # All exits are decisions at the close, filled next open.
                if stopped or age >= 2:
                    active = False
                    age = 0
                    armed = False
                else:
                    raw[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                age = 0
                armed = False
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.4, "low": 99.8,
         "close": 100.2, "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame(
        {"atr": 1.0, "confirmed": [False, False, True, False, False, False, False, False]},
        index=index
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    data.loc[index[3], ["high", "close"]] = [101.0, 100.7]
    # Breakeven armed on day one; day two touches the ratcheted stop.
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    data.loc[index[3], "low"] = 98.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    rng = np.random.default_rng(17)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 400)))
    history = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000.0},
        index=pd.date_range("2018-01-01", periods=400)
    )
    full = strategy.indicators(history, params)
    prefix = strategy.indicators(history.iloc[:320], params)
    pd.testing.assert_frame_equal(full.iloc[:320], prefix)
    assert not full["confirmed"].iloc[:strategy.warmup_bars(params)].any()


if __name__ == "__main__":
    _self_check()
