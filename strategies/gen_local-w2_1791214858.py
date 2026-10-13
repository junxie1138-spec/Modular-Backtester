from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEchoParams:
    memory: int = 63
    echo_lag: int = 5
    forecast_threshold: float = 0.20
    pressure_threshold: float = 0.15
    atr_period: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 2.5
    max_hold: int = 8

    def __post_init__(self):
        if min(self.memory, self.echo_lag, self.atr_period, self.max_hold) < 1 or self.memory < 3:
            raise ValueError("Lookbacks and holding period must be positive; memory must be >= 3.")
        if not all(np.isfinite(x) and x > 0 for x in (
            self.forecast_threshold, self.pressure_threshold,
            self.breakeven_pct, self.trail_atr,
        )):
            raise ValueError("Thresholds and stop parameters must be finite and positive.")


class GeneratedStrategy(BaseStrategy[GapEchoParams]):
    strategy_id = "gen_local-w2_1791214858"

    @classmethod
    def params_type(cls):
        return GapEchoParams

    def warmup_bars(self, params: GapEchoParams) -> int:
        return int(max(params.memory + params.echo_lag + 1, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GapEchoParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = data["open"] / previous_close - 1.0
        out["lagged_gap"] = out["gap"].shift(params.echo_lag)
        out["gap_corr"] = out["gap"].rolling(params.memory).corr(out["lagged_gap"])
        out["gap_scale"] = out["gap"].rolling(params.memory).std(ddof=0).replace(0, np.nan)
        out["gap_sum"] = out["gap"].rolling(params.echo_lag).sum()
        # Forecast the next block's overnight pressure from lag-L gap autocorrelation.
        out["forecast"] = (
            out["gap_corr"] * out["gap_sum"]
            / (out["gap_scale"] * np.sqrt(params.echo_lag))
        )
        span = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["body"] = ((data["close"] - data["open"]) / span).clip(-1.0, 1.0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["weighted_body"] = out["body"] * volume
        out["volume_sum"] = volume.rolling(params.echo_lag).sum().replace(0, np.nan)
        out["body_sum"] = out["weighted_body"].rolling(params.echo_lag).sum()
        out["pressure"] = out["body_sum"] / out["volume_sum"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean().where(lambda s: s > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapEchoParams,
    ) -> SignalFrame:
        forecast = indicators["forecast"]
        pressure = indicators["pressure"]
        candidates = np.select([
            (forecast > params.forecast_threshold) & (pressure > params.pressure_threshold),
            (forecast < -params.forecast_threshold) & (pressure < -params.pressure_threshold),
        ], [1, -1], default=0)
        valid = np.isfinite(indicators[["forecast", "pressure", "atr"]]).all(axis=1).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry, peak, stop = np.nan, np.nan, np.nan
        armed = False
        for i in range(len(data)):
            if side:
                if age == 0:
                    # Prior close's decision enters at this bar's observed open.
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        continue
                    peak = side * entry
                    stop = peak - params.trail_atr * atrs[i - 1]
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atrs[i]):
                    side = 0
                    continue
                oriented_close = side * closes[i]
                peak = max(peak, oriented_close)
                armed = armed or (peak - side * entry >= params.breakeven_pct * entry)
                if armed:
                    # Signed-price stop always rises; a short's actual stop price falls.
                    stop = max(stop, side * entry, peak - params.trail_atr * atrs[i])
                if oriented_close <= stop or age >= params.max_hold:
                    side = 0
                    continue
                targets[i] = side
            elif valid[i] and candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(candidates[i])
                age = 0
                armed = False
                targets[i] = side
        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapEchoParams()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for side in (1, -1):
        closes = 100.0 + side * np.array([0, 0, 2, 3, 2, 0, 0, 0])
        data = pd.DataFrame({
            "open": 100.0, "high": np.maximum(100.0, closes) + 0.5,
            "low": np.minimum(100.0, closes) - 0.5, "close": closes, "volume": 1000.0,
        }, index=index)
        derived = pd.DataFrame({
            "forecast": [side, 0, 0, 0, 0, 0, 0, 0],
            "pressure": [side, 0, 0, 0, 0, 0, 0, 0],
            "atr": [1, 1, 1, 1, 10, 10, 1, 1],
        }, index=index)
        signals = strategy.generate_signals(data, derived, ctx, params).data
        assert signals["signal"].tolist() == [0, side, side, side, side, side, 0, 0]
        assert (signals["size"] > 0).all()
        data["close"] = 100.0
        timed = strategy.generate_signals(data, derived, ctx, GapEchoParams(max_hold=2)).data
        assert timed["signal"].tolist() == [0, side, side, 0, 0, 0, 0, 0]
    rng = np.random.default_rng(42)
    index = pd.date_range("2021-01-01", periods=240, freq="B")
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    opens = np.r_[100.0, closes[:-1]] * np.exp(rng.normal(0, 0.004, len(index)))
    data = pd.DataFrame({
        "open": opens, "high": np.maximum(opens, closes) + 0.5,
        "low": np.minimum(opens, closes) - 0.5, "close": closes, "volume": 1000.0,
    }, index=index)
    full_indicators = strategy.indicators(data, params)
    full = strategy.generate_signals(data, full_indicators, ctx, params).data
    prefix = data.iloc[:180]
    prefix_indicators = strategy.indicators(prefix, params)
    pd.testing.assert_frame_equal(full_indicators.iloc[:180], prefix_indicators)
    pd.testing.assert_frame_equal(full.iloc[:180], strategy.generate_signals(prefix, prefix_indicators, ctx, params).data)
    assert set(full["signal"].unique()) <= {-1, 0, 1}


if __name__ == "__main__":
    _self_check()
