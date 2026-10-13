from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 63
    energy_window: int = 4
    energy_memory: int = 20
    atr_window: int = 14
    min_drawdown: float = 0.035
    energy_ceiling: float = 0.60
    min_repair: float = 0.01
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    initial_stop_atr: float = 3.0
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1790980890"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window,
                       16 + params.energy_window + params.energy_memory - 1,
                       params.atr_window + 1, params.energy_memory, 4))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.peak_window, params.energy_window, params.energy_memory,
               params.atr_window, params.max_hold) < 1:
            raise ValueError("Window lengths and max_hold must be positive.")
        if not (0 < params.energy_ceiling < 1 and 0 < params.min_drawdown < 1
                and params.min_repair > 0 and params.breakeven_pct > 0
                and params.trail_atr > 0 and params.initial_stop_atr > 0):
            raise ValueError("Invalid recovery or stop parameters.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        log_price = np.log(close)
        for scale in (1, 2, 4, 8):
            mean = log_price.rolling(scale, min_periods=scale).mean()
            detail = (mean - mean.shift(scale)) * np.sqrt(scale / 2.0)
            energy = detail.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window).mean()
            peak = energy.rolling(
                params.energy_memory, min_periods=params.energy_memory).max().shift(1)
            out[f"haar_{scale}"] = detail
            out[f"energy_{scale}"] = energy
            out[f"energy_peak_{scale}"] = peak
            out[f"contraction_{scale}"] = energy / peak.where(peak > 0)
        ratios = out[[f"contraction_{s}" for s in (1, 2, 4, 8)]]
        out["worst_contraction"] = ratios.max(axis=1, skipna=False)
        out["drawdown"] = close / close.rolling(
            params.peak_window, min_periods=params.peak_window).max() - 1.0
        out["repair"] = close / close.rolling(
            params.energy_memory, min_periods=params.energy_memory).min() - 1.0
        out["recovery_move"] = close.diff(3)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        ready = out[["worst_contraction", "drawdown", "repair",
                     "recovery_move", "atr"]].notna().all(axis=1)
        out["setup"] = (ready
                        & (out["drawdown"] <= -params.min_drawdown)
                        & (out["worst_contraction"] < params.energy_ceiling)
                        & (out["repair"] >= params.min_repair)
                        & (out["recovery_move"] > 0)
                        & (out["atr"] > 0))
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        active = pending = armed = False
        entry = stop = highest = 0.0
        age = 0
        for i in range(len(data)):
            valid = (np.isfinite([opens[i], highs[i], lows[i], atr[i]]).all()
                     and min(opens[i], highs[i], lows[i], atr[i]) > 0)
            if pending:
                pending = False
                if valid and i > 0 and np.isfinite(atr[i - 1]):
                    # Yesterday's decision fills at today's open.
                    active = True
                    armed = False
                    entry = highest = opens[i]
                    stop = entry - params.initial_stop_atr * atr[i - 1]
                    age = 0
            if active:
                age += 1
                # Only the stop known before this bar can trigger an exit.
                # An exit decision executes at the following bar's open.
                if (valid and lows[i] <= stop) or age >= params.max_hold:
                    active = False
                    continue
                if valid:
                    highest = max(highest, highs[i])
                    if highest >= entry * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, highest - params.trail_atr * atr[i])
                signals[i] = 1
            elif valid and confirmed[i] and i >= self.warmup_bars(params) - 1:
                pending = True
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RecoveryParams(peak_window=2, energy_window=1, energy_memory=1,
                            atr_window=1, max_hold=20)
    strategy = GeneratedStrategy()
    n = strategy.warmup_bars(params) + 6
    index = pd.date_range("2020-01-01", periods=n)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    start = strategy.warmup_bars(params) - 1
    ind = pd.DataFrame({"confirmed": False, "atr": 1.0}, index=index)
    ind.iloc[start, ind.columns.get_loc("confirmed")] = True
    data.iloc[start + 2, data.columns.get_loc("high")] = 104.0
    data.iloc[start + 3, data.columns.get_loc("low")] = 100.5
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[start:start + 5].tolist() == [0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    actual = strategy.indicators(data, params)
    assert not actual["confirmed"].any()
    prefix = strategy.indicators(data.iloc[:-2], params)
    pd.testing.assert_frame_equal(actual.iloc[:-2], prefix)


if __name__ == "__main__":
    _self_check()
