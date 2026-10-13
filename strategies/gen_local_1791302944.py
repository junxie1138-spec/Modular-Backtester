from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 40
    burst_threshold: float = 2.0
    atr_window: int = 14
    stop_atr: float = 1.5
    hold_bars: int = 2
    cooldown_bars: int = 2
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791302944"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        if params.energy_window < 2 or params.atr_window < 1:
            raise ValueError("Lookbacks must be positive; energy_window must be at least two.")
        if not (np.isfinite(params.burst_threshold) and params.burst_threshold > 1):
            raise ValueError("burst_threshold must exceed one.")
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("stop_atr must be positive.")
        if params.hold_bars not in (1, 2) or params.cooldown_bars < 0:
            raise ValueError("hold_bars must be one or two; cooldown_bars must be nonnegative.")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")
        return int(max(params.energy_window + 9, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        for half in (1, 2, 4):
            recent = out["log_return"].rolling(half, min_periods=half).sum()
            detail = (recent - recent.shift(half)) / np.sqrt(2.0 * half)
            out[f"detail_{half}"] = detail
            energy = detail.pow(2)
            out[f"energy_{half}"] = energy
            baseline = energy.shift(1).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            out[f"baseline_{half}"] = baseline
            out[f"ratio_{half}"] = energy / baseline.clip(lower=1e-16)
        ratios = out[["ratio_1", "ratio_2", "ratio_4"]]
        out["burst_strength"] = ratios.min(axis=1, skipna=False)
        burst = out["burst_strength"].ge(params.burst_threshold)
        out["event"] = burst & ~burst.shift(1, fill_value=False)
        counterphase = (
            (out["detail_1"] * out["detail_4"] < 0)
            & (out["log_return"] * out["detail_4"] < 0)
        )
        out["direction"] = np.sign(out["detail_4"]).where(counterphase, 0).fillna(0)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        count = len(data)
        raw = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        position = 0
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        trade_size = params.min_size
        next_entry = self.warmup_bars(params) - 1
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        directions = indicators["direction"].fillna(0).to_numpy(dtype=float)
        events = indicators["event"].fillna(False).to_numpy(dtype=bool)
        strengths = indicators["burst_strength"].to_numpy(dtype=float)
        for i in range(count):
            if position:
                if i == entry_bar:
                    # Freeze the stop around the next-open fill and decision-bar ATR.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                valid = np.isfinite(closes[i]) and np.isfinite(stop)
                stopped = valid and position * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if not valid or stopped or expired:
                    position = 0
                    next_entry = i + params.cooldown_bars + 1
                else:
                    raw[i] = position
                    sizes[i] = trade_size
                continue
            if (
                i >= next_entry and events[i] and directions[i] in (-1, 1)
                and np.isfinite(strengths[i]) and np.isfinite(atrs[i])
                and atrs[i] > 0 and np.isfinite(closes[i])
            ):
                position = int(directions[i])
                entry_bar = i + 1
                entry_atr = atrs[i]
                conviction = float(np.clip(
                    strengths[i] / params.burst_threshold - 1.0, 0.0, 1.0
                ))
                trade_size = params.min_size + (
                    params.max_size - params.min_size
                ) * conviction
                raw[i] = position
                sizes[i] = trade_size
        df = pd.DataFrame({"signal": raw, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletParams(energy_window=2, atr_window=1, cooldown_bars=0)
    index = pd.date_range("2020-01-01", periods=18)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert strategy.warmup_bars(params) == 11
    derived["event"] = False
    derived["direction"] = 0.0
    derived["burst_strength"] = 4.0
    derived["atr"] = 1.0
    derived.loc[index[[10, 13]], "event"] = True
    derived.loc[index[10], "direction"] = 1.0
    derived.loc[index[13], "direction"] = -1.0
    data.loc[index[11], "close"] = 98.0
    data.loc[index[14], "close"] = 102.0
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].iloc[10:16].tolist() == [0, 1, 0, 0, -1, 0]
    assert result["size"].iloc[11] == params.max_size
    data["close"] = 100.0
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].iloc[10:17].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
