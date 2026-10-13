from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    breakout_bars: int = 12
    energy_bars: int = 4
    baseline_bars: int = 24
    atr_bars: int = 14
    energy_threshold: float = 1.3
    stop_atr: float = 1.8
    hold_bars: int = 4
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791337224"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(8 + params.energy_bars - 1 + params.baseline_bars,
                       params.atr_bars + 1, params.breakout_bars + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if min(params.breakout_bars, params.energy_bars,
               params.baseline_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5.")
        if not (np.isfinite(params.energy_threshold) and params.energy_threshold > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0
                and np.isfinite(params.entry_slippage_bps)
                and 0 <= params.entry_slippage_bps < 10000):
            raise ValueError("Invalid energy threshold, stop distance, or slippage.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (1, 2, 4):
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            # Causal Haar detail: adjacent blocks, support lengths 2, 4, 8.
            out[f"detail_{half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{half}"] = out[f"detail_{half}"].pow(2).rolling(
                params.energy_bars, min_periods=params.energy_bars).mean()
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(
            axis=1, min_count=3)
        out["baseline"] = out["total_energy"].rolling(
            params.baseline_bars, min_periods=params.baseline_bars).mean().shift(1)
        out["energy_ratio"] = out["total_energy"] / out["baseline"].where(out["baseline"] > 0)
        out["coarse_share"] = out["energy_4"] / out["total_energy"].where(out["total_energy"] > 0)
        out["share_change"] = out["coarse_share"] - out["coarse_share"].shift(2)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_bars, min_periods=params.atr_bars).mean()
        out["upper"] = data["high"].rolling(params.breakout_bars).max().shift(1)
        out["lower"] = data["low"].rolling(params.breakout_bars).min().shift(1)
        out["conviction"] = (
            0.25 + 0.5 * (out["energy_ratio"] / params.energy_threshold - 1.0)
            + 0.25 * out["coarse_share"]
        ).clip(0.25, 0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletParams,
    ) -> SignalFrame:
        ready = np.isfinite(indicators).all(axis=1) & (indicators["atr"] > 0)
        shock = (ready
                 & (indicators["energy_ratio"] > params.energy_threshold)
                 & (indicators["share_change"] > 0)
                 & (indicators["energy_4"] > indicators["energy_1"]))
        long_entry = (shock & (data["close"] > indicators["upper"])
                      & (indicators["detail_2"] > 0) & (indicators["detail_4"] > 0))
        short_entry = (shock & (data["close"] < indicators["lower"])
                       & (indicators["detail_2"] < 0) & (indicators["detail_4"] < 0))
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        convictions = indicators["conviction"].fillna(0.25).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        side, entry_bar = 0, -1
        stop, distance, held_size = np.nan, 0.0, 0.25
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # Entry open is observed only on its own bar; ATR was frozen at decision time.
                    entry_price = opens[i] * (1.0 + side * params.entry_slippage_bps / 10000.0)
                    stop = entry_price - side * distance
                stopped = np.isfinite(stop) and side * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired or not np.isfinite(closes[i]) or not np.isfinite(stop):
                    side = 0
                # An exit stays flat for this decision; no same-bar re-entry.
            elif i >= self.warmup_bars(params) - 1 and entries[i]:
                side = int(entries[i])
                entry_bar = i + 1
                distance = params.stop_atr * atrs[i]
                held_size = float(convictions[i])
                stop = np.nan
            signals[i] = side
            sizes[i] = held_size if side else 0.25
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), WaveletParams()
    n = strategy.warmup_bars(params) + 12
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert flat["signal"].eq(0).all() and flat["size"].gt(0).all()
    decision = strategy.warmup_bars(params) - 1
    for side in (1, -1):
        for stop_exit in (False, True):
            sample = data.copy()
            sample.iloc[decision, sample.columns.get_loc("close")] = 100.0 + 2.0 * side
            if stop_exit:
                sample.iloc[decision + 1, sample.columns.get_loc("close")] = 100.0 - 4.0 * side
            ind = strategy.indicators(data, params).fillna(0.0)
            ind["atr"], ind["upper"], ind["lower"] = 1.0, 101.0, 99.0
            ind["energy_4"], ind["energy_1"] = 2.0, 1.0
            ind["detail_2"], ind["detail_4"] = float(side), float(side)
            ind["conviction"] = 0.7
            ind.loc[index[decision], ["energy_ratio", "share_change"]] = [2.0, 0.2]
            result = strategy.generate_signals(sample, ind, ctx, params).data
            length = 1 if stop_exit else params.hold_bars
            assert result["signal"].iloc[:decision + 1].eq(0).all()
            assert result["signal"].iloc[decision + 1:decision + 1 + length].eq(side).all()
            assert result["signal"].iloc[decision + 1 + length:].eq(0).all()
            assert result["size"].iloc[decision + 1:decision + 1 + length].eq(0.7).all()


if __name__ == "__main__":
    _self_check()
