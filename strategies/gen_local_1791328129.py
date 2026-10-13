from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletCompressionParams:
    percentile_window: int = 126
    energy_window: int = 5
    compression_quantile: float = 0.25
    concentration_quantile: float = 0.80
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletCompressionParams]):
    strategy_id = "gen_local_1791328129"

    @classmethod
    def params_type(cls):
        return WaveletCompressionParams

    def warmup_bars(self, params: WaveletCompressionParams) -> int:
        return int(max(32 + params.energy_window + params.percentile_window + 5,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame,
                   params: WaveletCompressionParams) -> pd.DataFrame:
        if (params.percentile_window < 2 or params.energy_window < 1
                or params.atr_window < 1 or params.hold_bars < 1
                or not 0 < params.compression_quantile < 1
                or not 0 < params.concentration_quantile < 1
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (1, 2, 4, 8, 16):
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window).mean()
        energy_cols = [f"energy_{half}" for half in (1, 2, 4, 8, 16)]
        out["total_energy"] = out[energy_cols].sum(axis=1, min_count=5)
        out["coarse_share"] = (out["energy_8"] + out["energy_16"]) / out[
            "total_energy"].where(out["total_energy"] > 0)
        out["log_range"] = np.log(
            data["high"].where(data["high"] > 0) /
            data["low"].where(data["low"] > 0))
        out["range_mean"] = out["log_range"].rolling(5, min_periods=5).mean()
        for source, target, quantile in (
            ("total_energy", "energy_cutoff", params.compression_quantile),
            ("range_mean", "range_cutoff", params.compression_quantile),
            ("coarse_share", "share_cutoff", params.concentration_quantile),
        ):
            out[target] = out[source].shift(1).rolling(
                params.percentile_window,
                min_periods=params.percentile_window).quantile(quantile)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["compressed"] = (
            (out["total_energy"] < out["energy_cutoff"]) &
            (out["range_mean"] < out["range_cutoff"]))
        out["recent_compression"] = out["compressed"].astype(float).rolling(
            5, min_periods=5).max().shift(1).fillna(0).astype(bool)
        out["concentration_cross"] = (
            (out["coarse_share"] > out["share_cutoff"]) &
            (out["coarse_share"].shift(1) <= out["share_cutoff"].shift(1)))
        out["ready"] = np.isfinite(out[
            ["haar_8", "haar_16", "energy_cutoff", "range_cutoff",
             "share_cutoff", "atr"]]).all(axis=1) & (out["atr"] > 0)
        out["entry"] = np.where(
            out["ready"] & out["recent_compression"] &
            out["concentration_cross"] &
            (out["range_mean"] < out["range_cutoff"]) &
            (out["haar_8"] * out["haar_16"] > 0),
            np.sign(out["haar_16"]).fillna(0), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: WaveletCompressionParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        stop = np.nan
        for i in range(len(data)):
            exited = False
            if position and i == entry_bar:
                # The previous close's decision fills at this bar's open.
                if (not np.isfinite(opening[i]) or opening[i] <= 0
                        or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0):
                    position = 0
                    exited = True
                else:
                    stop = opening[i] - position * params.stop_atr * atr[i - 1]
            if position:
                stopped = (np.isfinite(close[i]) and
                           position * (close[i] - stop) < 0)
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired or not np.isfinite(close[i]):
                    position = 0
                    exited = True
            if not position and not exited and i + 1 < len(data):
                if (entries[i] in (-1, 1) and np.isfinite(close[i])
                        and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                    position = int(entries[i])
                    entry_bar = i + 1
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Checks next-open anchoring, both fixed stops, expiry, and the exact shift.
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    data = pd.DataFrame(
        {"open": 105.0, "high": 110.0, "low": 95.0,
         "close": 105.0, "volume": 1000.0}, index=index)
    params = WaveletCompressionParams(stop_atr=2.0)
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        frame = pd.DataFrame({"atr": 1.0, "entry": 0}, index=index)
        frame.iloc[0, frame.columns.get_loc("entry")] = direction
        prices = data.copy()
        prices.iloc[0, prices.columns.get_loc("close")] = 100.0
        prices.iloc[4, prices.columns.get_loc("close")] = 105.0 - direction * 3.0
        signals = strategy.generate_signals(prices, frame, None, params).data
        assert signals["signal"].iloc[0] == 0
        assert (signals["signal"].iloc[1:5] == direction).all()
        assert (signals["signal"].iloc[5:] == 0).all()
        assert (signals["size"] > 0).all()
    frame = pd.DataFrame({"atr": 1.0, "entry": 0}, index=index)
    frame.iloc[0, frame.columns.get_loc("entry")] = 1
    signals = strategy.generate_signals(data, frame, None, params).data
    assert (signals["signal"].iloc[1:19] == 1).all()
    assert (signals["signal"].iloc[19:] == 0).all()
