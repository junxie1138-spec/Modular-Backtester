from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    energy_window: int = 12
    fine_share_min: float = 0.65
    stretch_min: float = 1.1
    atr_period: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 4
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local-w2_1790977872"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(params.energy_window + 7, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if params.energy_window < 2 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive; energy_window must be >= 2.")
        if not 0 < params.fine_share_min < 1 or params.stretch_min <= 0:
            raise ValueError("Invalid wavelet thresholds.")
        if params.stop_atr <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Use a positive stop and a 3-5 session holding limit.")
        if not 0 <= params.entry_slippage_bps < 10000:
            raise ValueError("Invalid entry slippage.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["mean2"] = out["log_close"].rolling(2, min_periods=2).mean()
        out["mean4"] = out["log_close"].rolling(4, min_periods=4).mean()
        # Causal, undecimated Haar contrasts of adjacent equal-length blocks.
        out["wave2"] = (out["log_close"] - out["log_close"].shift(1)) / np.sqrt(2.0)
        out["wave4"] = out["mean2"] - out["mean2"].shift(2)
        out["wave8"] = np.sqrt(2.0) * (out["mean4"] - out["mean4"].shift(4))
        for scale in (2, 4, 8):
            out[f"energy{scale}"] = out[f"wave{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out["energy2"] + out["energy4"] + out["energy8"]
        out["fine_share"] = (out["energy2"] + out["energy4"]) / out["total_energy"].where(
            out["total_energy"] > 0
        )
        out["stretch"] = out["wave4"].abs() / np.sqrt(
            out["energy4"].where(out["energy4"] > 0)
        )
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["opposition"] = out["wave2"] * out["wave4"]
        out = out.replace([np.inf, -np.inf], np.nan)
        eligible = (
            (out["fine_share"] >= params.fine_share_min)
            & (out["stretch"] >= params.stretch_min)
            & (out["opposition"] < 0)
            & (out["atr"] > 0)
        ).fillna(False)
        out["candidate"] = np.where(eligible, -np.sign(out["wave4"].fillna(0)), 0).astype(int)
        out["conviction"] = (
            (out["stretch"] / params.stretch_min - 1.0)
            + (out["fine_share"] - params.fine_share_min) / (1.0 - params.fine_share_min)
        ).clip(0.0, 1.0).fillna(0.0)
        out["entry_size"] = 0.25 + 0.75 * out["conviction"]
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].fillna(0.25).to_numpy(dtype=float)
        side, age = 0, 0
        stop, distance, allocation = np.nan, np.nan, 1.0
        for i in range(n):
            if side:
                if age == 0:
                    # Entry is this bar's open; ATR was frozen at the decision close.
                    entry = opens[i] * (1.0 + side * params.entry_slippage_bps / 10000.0)
                    stop = entry - side * distance
                age += 1
                invalid = not np.isfinite(closes[i]) or not np.isfinite(stop)
                stopped = side * (closes[i] - stop) < 0
                if invalid or stopped or age >= params.hold_bars:
                    side, age = 0, 0
                    continue
            elif candidates[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidates[i])
                age = 0
                distance = params.stop_atr * atr[i]
                allocation = float(np.clip(entry_sizes[i], 0.25, 1.0))
            signal[i] = side
            if side:
                size[i] = allocation
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), WaveParams()
    index = pd.date_range("2020-01-01", periods=7)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for side in (-1, 1):
        for hit_stop in (False, True):
            data = pd.DataFrame(
                {"open": 100.0, "high": 106.0, "low": 94.0,
                 "close": 100.0, "volume": 1000.0}, index=index
            )
            if hit_stop:
                data.iloc[2, data.columns.get_loc("close")] = 100.0 - side * 5.0
            indicators = pd.DataFrame(
                {"atr": [1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                 "candidate": [side, 0, 0, 0, 0, 0, 0],
                 "entry_size": 0.6}, index=index
            )
            result = strategy.generate_signals(data, indicators, ctx, params).data
            expected = [0, side, side, 0, 0, 0, 0] if hit_stop else [0, side, side, side, side, 0, 0]
            assert result["signal"].tolist() == expected
            assert result["size"].iloc[1] == 0.6
            assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
