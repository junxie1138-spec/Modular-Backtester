from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 16
    corr_window: int = 24
    atr_window: int = 14
    energy_gate: float = 0.48
    corr_gate: float = 0.10
    breakeven_pct: float = 0.004
    trail_atr: float = 0.8
    initial_stop_atr: float = 1.5
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790894830"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(params.energy_window + 8, params.corr_window + 2, params.atr_window + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if min(params.energy_window, params.corr_window, params.atr_window) < 2:
            raise ValueError("Windows must be at least two bars.")
        if not (0 < params.energy_gate < 1 and 0 < params.corr_gate < 1):
            raise ValueError("Energy and correlation gates must lie between zero and one.")
        if min(params.breakeven_pct, params.trail_atr, params.initial_stop_atr) <= 0:
            raise ValueError("Stop parameters must be positive.")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two bars.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        r = out["return"]
        # Causal, undecimated Haar details at two-, four-, and eight-bar scales.
        out["haar_2"] = (r - r.shift(1)) / np.sqrt(2.0)
        pair = r + r.shift(1)
        out["haar_4"] = (pair - pair.shift(2)) / 2.0
        quartet = pair + pair.shift(2)
        out["haar_8"] = (quartet - quartet.shift(4)) / np.sqrt(8.0)
        for scale in (2, 4, 8):
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(params.energy_window).mean()
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        out["fine_share"] = out["energy_2"] / denominator
        out["coarse_share"] = out["energy_8"] / denominator
        out["autocorr"] = r.rolling(params.corr_window).corr(r.shift(1)).clip(-1, 1)
        out["impulse"] = r.rolling(2).sum()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        persistent = (out["coarse_share"] >= params.energy_gate) & (out["autocorr"] >= params.corr_gate)
        elastic = (out["fine_share"] >= params.energy_gate) & (out["autocorr"] <= -params.corr_gate)
        direction = np.sign(out["impulse"]).fillna(0).astype(int)
        out["candidate"] = 0
        out.loc[persistent, "candidate"] = direction[persistent]
        out.loc[elastic, "candidate"] = -direction[elastic]
        ready = out[["fine_share", "coarse_share", "autocorr", "impulse", "atr"]].notna().all(axis=1)
        out.loc[~ready | (out["atr"] <= 0), "candidate"] = 0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        age = 0
        entry = np.nan
        entry_atr = np.nan
        stop = -np.inf
        armed = False
        for i in range(len(data)):
            if side:
                # The previous close's entry instruction fills at this open.
                if age == 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    entry = opens[i]
                    stop = side * entry - params.initial_stop_atr * entry_atr
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    side = 0
                    continue
                oriented_close = side * closes[i]
                # Stops are evaluated at the close; exits fill at the next open.
                if oriented_close <= stop or age >= params.max_hold_bars:
                    side = 0
                    continue
                if side * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed-price stops only rise: long stops rise, short stops fall.
                    stop = max(stop, side * entry, oriented_close - params.trail_atr * atr[i])
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(confirmed[i])
                age = 0
                entry_atr = atr[i]
                armed = False
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=64)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert flat["signal"].eq(0).all() and flat["size"].gt(0).all()
    forced = pd.DataFrame({"candidate": 1, "atr": 0.5}, index=index)
    long_signals = strategy.generate_signals(data, forced, ctx, params).data["signal"]
    assert long_signals.iloc[:5].tolist() == [0, 0, 1, 1, 0]
    forced["candidate"] = -1
    short_signals = strategy.generate_signals(data, forced, ctx, params).data["signal"]
    assert short_signals.iloc[:5].tolist() == [0, 0, -1, -1, 0]
    data.loc[index[2], "close"] = 101.0
    forced["candidate"] = 1
    params.max_hold_bars = 1
    one_day = strategy.generate_signals(data, forced, ctx, params).data["signal"]
    assert one_day.iloc[:4].tolist() == [0, 0, 1, 0]


if __name__ == "__main__":
    _self_check()
