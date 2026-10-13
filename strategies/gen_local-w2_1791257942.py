from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletAnalogParams:
    history_bars: int = 378
    neighbors: int = 7
    holding_bars: int = 18
    percentile_bars: int = 126
    entry_percentile: float = 0.8
    atr_bars: int = 14
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[WaveletAnalogParams]):
    strategy_id = "gen_local-w2_1791257942"

    @classmethod
    def params_type(cls) -> type[WaveletAnalogParams]:
        return WaveletAnalogParams

    def warmup_bars(self, params: WaveletAnalogParams) -> int:
        return int(max(params.history_bars + params.holding_bars + 64
                       + params.percentile_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletAnalogParams) -> pd.DataFrame:
        if (params.neighbors < 1 or params.holding_bars < 1
                or params.history_bars < params.neighbors * params.holding_bars
                or params.percentile_bars < 2 or params.atr_bars < 1
                or not 0.0 < params.entry_percentile < 1.0
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0):
            raise ValueError("Invalid wavelet analog parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        for scale in (2, 4, 8):
            block = out["log_return"].rolling(scale, min_periods=scale).mean()
            out[f"haar_{scale}"] = (block - block.shift(scale)) * np.sqrt(scale / 2.0)
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(16, min_periods=16).mean()
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        features = []
        for scale in (2, 4, 8):
            name = f"share_{scale}"
            out[name] = out[f"energy_{scale}"] / denominator
            features.append(name)
        out["phase"] = np.tanh(out["haar_8"] / np.sqrt(denominator))
        features.append("phase")
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        # Completed labels use next-open entry and next-open exit.
        entry_open = data["open"].shift(params.holding_bars).where(lambda x: x > 0)
        out["realized_return"] = data["open"] / entry_open - 1.0
        matrix = out[features].to_numpy(dtype=float)
        realized = out["realized_return"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        h = params.holding_bars
        # ponytail: linear scan over a bounded archive; index it only for intraday-scale histories.
        for i in range(len(data)):
            if not np.isfinite(matrix[i]).all():
                continue
            # Labels end at j+h+1; 32 extra bars embargo overlapping query shapes.
            end = i - h - 33
            start = max(31, end - params.history_bars + 1)
            candidates = np.arange(end, start - 1, -h, dtype=int)
            if len(candidates) < params.neighbors:
                continue
            labels = realized[candidates + h + 1]
            valid = np.isfinite(matrix[candidates]).all(axis=1) & np.isfinite(labels)
            candidates, labels = candidates[valid], labels[valid]
            if len(candidates) < params.neighbors:
                continue
            distance = np.square(matrix[candidates] - matrix[i]).sum(axis=1)
            nearest = np.argsort(distance, kind="stable")[:params.neighbors]
            forecast[i] = labels[nearest].mean()
        out["forecast"] = forecast
        out["entry_threshold"] = out["forecast"].abs().rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveletAnalogParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        threshold = indicators["entry_threshold"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position, entered = 0, -1
        high_water = low_water = stop = np.nan
        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = 0
                continue
            if position:
                if i == entered + 1:
                    high_water = low_water = close[i]
                    stop = close[i] - position * params.trail_atr * atr[i]
                high_water = max(high_water, close[i])
                low_water = min(low_water, close[i])
                if position == 1:
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    stopped = close[i] <= stop
                else:
                    stop = min(stop, low_water + params.trail_atr * atr[i])
                    stopped = close[i] >= stop
                if stopped or i - entered >= params.holding_bars:
                    position = 0
            elif (np.isfinite(forecast[i]) and np.isfinite(threshold[i])
                  and abs(forecast[i]) > threshold[i]):
                position = 1 if forecast[i] > 0 else -1
                entered = i
                high_water = low_water = close[i]
                stop = close[i] - position * params.trail_atr * atr[i]
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = WaveletAnalogParams()
    rng = np.random.default_rng(19)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 900)))
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000000.0
    }, index=pd.date_range("2000-01-01", periods=len(close), freq="B"))
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:700], params)
    pd.testing.assert_frame_equal(full.iloc[:700], prefix)
    signals = strategy.generate_signals(data, full, None, params).data
    earlier = strategy.generate_signals(data.iloc[:700], prefix, None, params).data
    pd.testing.assert_frame_equal(signals.iloc[:700], earlier)
    assert signals["signal"].isin([-1, 0, 1]).all()
    assert (signals["size"] > 0).all() and signals["signal"].iloc[0] == 0
    assert full["forecast"].notna().any()
    for direction, prices in ((1, [100., 105., 101., 99.]), (-1, [100., 95., 99., 101.])):
        sample = data.iloc[:4].copy()
        sample["close"] = prices
        inputs = pd.DataFrame({
            "atr": 1.0, "forecast": float(direction), "entry_threshold": 0.5
        }, index=sample.index)
        result = strategy.generate_signals(sample, inputs, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0]


if __name__ == "__main__":
    _self_check()
