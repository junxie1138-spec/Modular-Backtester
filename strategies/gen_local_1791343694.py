from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    drawdown_window: int = 63
    energy_window: int = 5
    pressure_window: int = 8
    atr_window: int = 14
    min_drawdown: float = 0.035
    entropy_delta: float = 0.035
    pressure_threshold: float = 0.12
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791343694"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.drawdown_window, 16 + params.energy_window + 3,
                       params.pressure_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        windows = (params.drawdown_window, params.energy_window,
                   params.pressure_window, params.atr_window, params.hold_bars)
        if any(w < 1 for w in windows):
            raise ValueError("Window lengths and hold_bars must be positive.")
        thresholds = (params.min_drawdown, params.entropy_delta,
                      params.pressure_threshold, params.trail_atr)
        if not all(np.isfinite(x) and x > 0 for x in thresholds):
            raise ValueError("Thresholds and trail_atr must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        out["rolling_high"] = close.rolling(params.drawdown_window,
                                             min_periods=params.drawdown_window).max()
        out["drawdown"] = 1.0 - close / out["rolling_high"]
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        energies = []
        for scale in (2, 4, 8, 16):
            half = scale // 2
            mean_name = f"return_mean_{half}"
            out[mean_name] = out["log_return"].rolling(half, min_periods=half).mean()
            out[f"haar_{scale}"] = np.sqrt(half / 2.0) * (
                out[mean_name] - out[mean_name].shift(half))
            energy_name = f"energy_{scale}"
            out[energy_name] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window).mean()
            energies.append(energy_name)
        out["total_energy"] = out[energies].sum(axis=1, min_count=4)
        entropy_terms = []
        for scale, energy_name in zip((2, 4, 8, 16), energies):
            share_name = f"energy_share_{scale}"
            term_name = f"entropy_term_{scale}"
            out[share_name] = out[energy_name] / out["total_energy"].where(
                out["total_energy"] > 0)
            out[term_name] = -out[share_name] * np.log(
                out[share_name].clip(lower=1e-15))
            entropy_terms.append(term_name)
        out["entropy"] = out[entropy_terms].sum(axis=1, min_count=4) / np.log(4.0)
        out["entropy_change"] = out["entropy"].diff(3)
        out["energy_change"] = out["total_energy"].diff(3)
        out["close_change"] = close.diff()
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0)
        out["signed_volume"] = np.sign(out["close_change"]) * out["valid_volume"]
        out["volume_sum"] = out["valid_volume"].rolling(
            params.pressure_window, min_periods=params.pressure_window).sum()
        out["signed_volume_sum"] = out["signed_volume"].rolling(
            params.pressure_window, min_periods=params.pressure_window).sum()
        out["volume_pressure"] = out["signed_volume_sum"] / out["volume_sum"].where(
            out["volume_sum"] > 0)
        out["stressed"] = out["drawdown"] >= params.min_drawdown
        out["wavelet_recovery"] = (
            (out["entropy_change"] >= params.entropy_delta) &
            (out["energy_change"] < 0))
        out["wavelet_relapse"] = (
            (out["entropy_change"] <= -params.entropy_delta) &
            (out["energy_change"] > 0))
        out["long_entry"] = (out["stressed"] & out["wavelet_recovery"] &
                              (out["volume_pressure"] >= params.pressure_threshold))
        out["short_entry"] = (out["stressed"] & out["wavelet_relapse"] &
                               (out["volume_pressure"] <= -params.pressure_threshold))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water = low_water = stop = np.nan
        for i in range(len(data)):
            price, volatility = close[i], atr[i]
            if not (np.isfinite(price) and price > 0 and
                    np.isfinite(volatility) and volatility > 0):
                position = 0
                high_water = low_water = stop = np.nan
                continue
            if position:
                age += 1
                # The first watermark is the close of the actual fill bar.
                high_water = price if np.isnan(high_water) else max(high_water, price)
                low_water = price if np.isnan(low_water) else min(low_water, price)
                if position == 1:
                    candidate = high_water - params.trail_atr * volatility
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    stopped = price <= stop
                else:
                    # Shorts use the symmetric lowest-close, downward-only trail.
                    candidate = low_water + params.trail_atr * volatility
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    stopped = price >= stop
                if stopped or age >= params.hold_bars:
                    position = 0
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                age = 0
                high_water = low_water = stop = np.nan
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 105., 101., 100., 96., 100., 100.]}, index=index)
    features = pd.DataFrame({
        "atr": 1.0,
        "long_entry": [True, False, False, False, False, False, False],
        "short_entry": [False, False, False, True, False, False, False]
    }, index=index)
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    params.hold_bars = 2
    data["close"] = 100.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    index = pd.date_range("2021-01-01", periods=120)
    close = 100.0 * np.exp(0.02 * np.sin(np.arange(120) / 5.0))
    bars = pd.DataFrame({"open": close, "high": close + 1.0,
                         "low": close - 1.0, "close": close,
                         "volume": 1000000.0}, index=index)
    params = StrategyParams()
    features = strategy.indicators(bars, params)
    cut = 90
    prefix = strategy.indicators(bars.iloc[:cut], params)
    pd.testing.assert_frame_equal(features.iloc[:cut], prefix)
    full = strategy.generate_signals(bars, features, None, params).data
    partial = strategy.generate_signals(bars.iloc[:cut], prefix, None, params).data
    pd.testing.assert_frame_equal(full.iloc[:cut], partial)
    assert strategy.warmup_bars(params) >= 16 + params.energy_window + 3
    assert set(full["signal"].unique()) <= {-1, 0, 1}


if __name__ == "__main__":
    _self_check()
