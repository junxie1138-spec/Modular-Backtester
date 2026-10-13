from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletConcentrationParams:
    energy_window: int = 10
    coarse_share_min: float = 0.40
    entropy_max: float = 0.85
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveletConcentrationParams]):
    strategy_id = "gen_local-w2_1791315821"

    @classmethod
    def params_type(cls):
        return WaveletConcentrationParams

    def warmup_bars(self, params):
        return int(max(params.energy_window + 10, params.atr_window + 1))

    def indicators(self, data, params):
        if (params.energy_window < 2 or params.atr_window < 1
                or not 0 < params.coarse_share_min < 1
                or not 0 < params.entropy_max <= 1
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0
                or not 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid wavelet, ATR, or holding-period parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        r = out["return"]
        for half in (1, 2, 4):
            mean = r.rolling(half, min_periods=half).mean()
            out[f"mean_{half}"] = mean
            out[f"detail_{half}"] = np.sqrt(half / 2.0) * (
                mean - mean.shift(half))
        out["coarse"] = np.sqrt(8.0) * r.rolling(8, min_periods=8).mean()
        energy_names = []
        for name in ("detail_1", "detail_2", "detail_4", "coarse"):
            energy_name = f"energy_{name}"
            out[energy_name] = out[name].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window).mean()
            energy_names.append(energy_name)
        out["total_energy"] = out[energy_names].sum(axis=1, min_count=4)
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        shares = out[energy_names].div(denominator, axis=0)
        for name in energy_names:
            out[f"share_{name}"] = shares[name]
        out["coarse_share"] = shares["energy_coarse"]
        out["entropy"] = -(shares * np.log(shares.clip(lower=1e-12))).sum(
            axis=1, min_count=4) / np.log(4.0)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        eligible = (
            (out["coarse_share"] >= params.coarse_share_min)
            & (out["entropy"] <= params.entropy_max)
            & (out["entropy"] < out["entropy"].shift(1))
            & out["atr"].gt(0)
            & np.isfinite(out["atr"])
            & np.isfinite(out["coarse"])
        )
        out["candidate"] = np.sign(out["coarse"]).where(
            eligible, 0).fillna(0).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if position:
                # Record the next-bar entry open only once that bar is observed.
                if i == entry_bar:
                    if (not np.isfinite(opens[i]) or opens[i] <= 0
                            or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0):
                        position = 0
                    else:
                        stop = opens[i] - position * params.stop_atr * atr[i - 1]
                if position:
                    stopped = (
                        not np.isfinite(closes[i])
                        or (position == 1 and closes[i] < stop)
                        or (position == -1 and closes[i] > stop)
                    )
                    if stopped or i - entry_bar + 1 >= params.hold_bars:
                        position = 0
            elif (confirmed[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0
                  and np.isfinite(closes[i]) and closes[i] > 0):
                position = int(confirmed[i])
                entry_bar = i + 1
                stop = np.nan
            raw[i] = position
        df["signal"] = raw
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=16)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0,
         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[3], "close"] = 97.0
    data.loc[index[7], "close"] = 103.0
    indicators = pd.DataFrame(
        {"atr": 1.0, "confirmed": [0, 1, 0, 0, 0, -1, 0, 0,
                                  0, 1, 0, 0, 0, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(
        data, indicators, None, WaveletConcentrationParams()).data
    assert result["signal"].tolist() == [
        0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
