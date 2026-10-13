from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    path_bars: int = 18
    forecast_bars: int = 18
    history_bars: int = 504
    neighbors: int = 8
    min_forecast: float = 0.004


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local-w2_1791320842"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(params.history_bars + params.forecast_bars + params.path_bars + 1)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        if (params.path_bars < 2 or params.forecast_bars < 1
                or params.neighbors < 1 or params.min_forecast < 0
                or not np.isfinite(params.min_forecast)
                or params.history_bars < params.neighbors * params.forecast_bars):
            raise ValueError("Invalid shape or analog parameters")
        out = pd.DataFrame(index=data.index)
        log_close = np.log(data["close"].where(data["close"] > 0))
        out["log_return"] = log_close.diff()
        out["volume_change"] = np.log1p(
            data["volume"].where(data["volume"] >= 0)
        ).diff()
        r = np.column_stack([
            out["log_return"].shift(lag).to_numpy()
            for lag in range(params.path_bars - 1, -1, -1)
        ])
        v = np.column_stack([
            out["volume_change"].shift(lag).to_numpy()
            for lag in range(params.path_bars - 1, -1, -1)
        ])
        # Antisymmetric second-order signature of cumulative price/volume increments.
        area = 0.5 * ((r.cumsum(axis=1) - r) * v
                      - (v.cumsum(axis=1) - v) * r).sum(axis=1)
        scale = np.abs(r).sum(axis=1) * np.abs(v).sum(axis=1)
        out["signature_area"] = np.divide(
            area, scale, out=np.full(len(data), np.nan), where=scale > 1e-12
        )
        spread = data["high"] - data["low"]
        pressure = (data["close"] - data["open"]) / spread.where(spread > 0)
        pressure = pressure.mask(spread == 0, 0.0).clip(-1.0, 1.0)
        out["candle_pressure"] = pressure
        out["pressure_shape"] = pressure.rolling(
            params.path_bars, min_periods=params.path_bars
        ).mean()
        out["matured_return"] = log_close.diff(params.forecast_bars)
        out = out.replace([np.inf, -np.inf], np.nan)
        features = out[["signature_area", "pressure_shape"]].to_numpy()
        realized = out["matured_return"].to_numpy()
        forecasts = np.full((len(data), 2), np.nan)
        # ponytail: bounded brute-force analog search; index only if history grows materially.
        for t in range(self.warmup_bars(params) - 1, len(data)):
            if not np.isfinite(features[t]).all():
                continue
            # Completed labels end before the current shape starts.
            stop = t - params.forecast_bars - params.path_bars
            candidates = np.arange(0, stop + 1, params.forecast_bars)
            candidates = candidates[candidates >= stop - params.history_bars + 1]
            labels = realized[candidates + params.forecast_bars]
            valid = np.isfinite(features[candidates]).all(axis=1) & np.isfinite(labels)
            candidates, labels = candidates[valid], labels[valid]
            if len(candidates) < params.neighbors:
                continue
            # Separate analog votes enforce two-primitive AND, rather than a blended score.
            for primitive in range(2):
                distance = np.abs(features[candidates, primitive] - features[t, primitive])
                selected = np.argsort(distance, kind="stable")[:params.neighbors]
                forecasts[t, primitive] = labels[selected].mean()
        out["area_forecast"] = forecasts[:, 0]
        out["pressure_forecast"] = forecasts[:, 1]
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ShapeParams,
    ) -> SignalFrame:
        forecasts = indicators[["area_forecast", "pressure_forecast"]].to_numpy()
        valid = np.isfinite(forecasts).all(axis=1)
        long_entry = valid & (forecasts > params.min_forecast).all(axis=1)
        short_entry = valid & (forecasts < -params.min_forecast).all(axis=1)
        entry = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for t, direction in enumerate(entry):
            if position and direction == -position:
                position = 0  # Opposite agreement exits; always emit a flat bar.
            elif position == 0 and direction:
                position = int(direction)
            signal[t] = position  # Neutral or missing forecasts preserve an existing trade.
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ShapeParams(path_bars=6, forecast_bars=15, history_bars=120, neighbors=4)
    rng = np.random.default_rng(23)
    index = pd.date_range("2020-01-01", periods=260, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index))))
    opened = close * np.exp(rng.normal(0.0, 0.003, len(index)))
    data = pd.DataFrame({
        "open": opened, "high": np.maximum(opened, close) * 1.01,
        "low": np.minimum(opened, close) * 0.99, "close": close,
        "volume": rng.integers(100000, 900000, len(index)).astype(float),
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:220], params)
    pd.testing.assert_frame_equal(full.iloc[:220], prefix)
    assert full["area_forecast"].notna().any()
    result = strategy.generate_signals(data, full, ctx, params).data
    assert result.index.equals(data.index)
    assert result["signal"].isin([-1, 0, 1]).all()
    assert (result["size"] > 0).all()
    assert (result["signal"].iloc[:strategy.warmup_bars(params)] == 0).all()
    votes = pd.DataFrame({
        "area_forecast": [0.02, np.nan, -0.02, -0.02, 0.02, 0.02],
        "pressure_forecast": [0.02, np.nan, -0.02, -0.02, -0.02, 0.02],
    }, index=index[:6])
    actual = strategy.generate_signals(data.iloc[:6], votes, ctx, params).data
    assert actual["signal"].tolist() == [0, 1, 1, 0, -1, -1]


if __name__ == "__main__":
    _self_check()
