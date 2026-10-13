from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    frac_order: float = 0.45
    frac_window: int = 64
    center_window: int = 32
    corr_window: int = 60
    half_cycle: int = 18
    resonance: float = 0.15
    hold_bars: int = 18
    atr_window: int = 14
    vol_window: int = 20
    target_vol: float = 0.12
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1791272351"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(
            200,
            params.frac_window + params.center_window
            + 2 * params.half_cycle + params.corr_window - 2,
            params.vol_window + 1,
            params.atr_window + 1,
        ) + 1)

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if not 0.0 < params.frac_order < 1.0:
            raise ValueError("frac_order must lie between zero and one")
        if min(params.frac_window, params.center_window, params.corr_window,
               params.half_cycle, params.hold_bars, params.atr_window,
               params.vol_window) < 2:
            raise ValueError("windows and hold_bars must be at least two")
        if not 0.0 < params.resonance < 1.0:
            raise ValueError("resonance must lie between zero and one")
        if params.target_vol <= 0.0 or params.stop_atr <= 0.0:
            raise ValueError("target_vol and stop_atr must be positive")
        close = data["close"].astype(float)
        weights = np.empty(params.frac_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        frac = close * weights[0]
        # ponytail: finite fractional memory; increase frac_window for longer memory.
        for k in range(1, params.frac_window):
            frac = frac + weights[k] * close.shift(k)
        out = pd.DataFrame(index=data.index)
        out["frac"] = frac
        out["equilibrium"] = frac.rolling(params.center_window).mean()
        out["dispersion"] = frac.rolling(params.center_window).std(ddof=0)
        out["z"] = (frac - out["equilibrium"]) / out["dispersion"].replace(0.0, np.nan)
        out["half_corr"] = out["z"].rolling(params.corr_window).corr(
            out["z"].shift(params.half_cycle))
        out["full_corr"] = out["z"].rolling(params.corr_window).corr(
            out["z"].shift(2 * params.half_cycle))
        out["ma200"] = close.rolling(200).mean()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["returns"] = close.pct_change(fill_method=None)
        out["annual_vol"] = out["returns"].rolling(params.vol_window).std(
            ddof=0) * np.sqrt(252.0)
        out["previous_z"] = out["z"].shift(1)
        out["resonant"] = (
            (out["half_corr"] < -params.resonance)
            & (out["full_corr"] > params.resonance))
        out["entry"] = np.select([
            out["resonant"] & (close > out["ma200"])
            & (out["z"] > 0.0) & (out["previous_z"] <= 0.0),
            out["resonant"] & (close < out["ma200"])
            & (out["z"] < 0.0) & (out["previous_z"] >= 0.0),
        ], [1, -1], default=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveParams,
    ) -> SignalFrame:
        n = len(data)
        decisions = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        vol = indicators["annual_vol"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(0).to_numpy(dtype=int)
        ma = indicators["ma200"].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        stop = np.nan
        exposure = 1.0
        pending_atr = np.nan
        for i in range(n):
            if side:
                # The prior close's decision becomes an entry at today's open.
                if entry_bar < 0:
                    if not np.isfinite(opening[i]) or opening[i] <= 0.0:
                        side = 0
                        continue
                    entry_bar = i
                    stop = opening[i] - side * params.stop_atr * pending_atr
                stopped = (
                    not np.isfinite(close[i])
                    or side * (close[i] - stop) < 0.0)
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    side = 0
                    entry_bar = -1
                    continue
            elif (
                i >= self.warmup_bars(params) - 1
                and entry[i] != 0
                and np.isfinite(close[i]) and close[i] > 0.0
                and np.isfinite(ma[i])
                and np.isfinite(atr[i]) and atr[i] > 0.0
                and np.isfinite(vol[i]) and vol[i] > 0.0
            ):
                side = int(entry[i])
                pending_atr = atr[i]
                exposure = float(np.clip(params.target_vol / vol[i], 0.01, 1.0))
                entry_bar = -1
            decisions[i] = side
            if side:
                sizes[i] = exposure
        df = pd.DataFrame({"signal": decisions, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveParams()
    index = pd.date_range("2020-01-01", periods=220, freq="B")
    data = pd.DataFrame({
        "open": 101.0, "high": 102.0, "low": 94.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": 0, "atr": 2.0, "annual_vol": 0.2, "ma200": 90.0,
    }, index=index)
    indicators.loc[index[200], "entry"] = 1
    data.loc[index[202], "close"] = 96.0
    data.loc[index[203], "close"] = 95.0
    indicators.loc[index[202]:, "atr"] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[200:205].tolist() == [0, 1, 1, 1, 0]
    assert np.isclose(result["size"].iloc[201], 0.6)
    indicators["entry"] = 0
    indicators.loc[index[200], "entry"] = -1
    indicators["atr"] = 2.0
    indicators["ma200"] = 110.0
    data["close"] = 100.0
    data.loc[index[203], "close"] = 107.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[200:205].tolist() == [0, -1, -1, -1, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert (result["signal"].iloc[201:219] == -1).all()
    assert result["signal"].iloc[219] == 0
    assert (result["size"] > 0.0).all()
    assert set(result["signal"].unique()) <= {-1, 0, 1}


if __name__ == "__main__":
    _self_check()
