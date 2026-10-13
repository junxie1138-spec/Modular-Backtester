from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    streak_len: int = 3
    baseline_bars: int = 30
    close_threshold: float = 0.65
    drought_fraction: float = 0.25
    flow_threshold: float = 0.15
    atr_bars: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = "gen_local_1791234574"

    @classmethod
    def params_type(cls):
        return PressureParams

    def warmup_bars(self, params: PressureParams) -> int:
        return int(max(params.baseline_bars + params.streak_len + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: PressureParams) -> pd.DataFrame:
        if not (params.streak_len >= 2 and params.baseline_bars >= 2
                and params.atr_bars >= 1 and params.hold_bars >= 1
                and 0.5 < params.close_threshold < 1.0
                and 0.0 <= params.drought_fraction < 0.5
                and 0.0 < params.flow_threshold < 1.0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Invalid pressure strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["close_location"] = (data["close"] - data["low"]) / out["range"]
        out["body"] = ((data["close"] - data["open"]) / out["range"]).clip(-1.0, 1.0)
        out["valid_volume"] = data["volume"].where(data["volume"] > 0)
        out["signed_volume"] = out["body"] * out["valid_volume"]
        for label, flag in (
            ("up", out["close_location"].ge(params.close_threshold)),
            ("down", out["close_location"].le(1.0 - params.close_threshold)),
        ):
            out[label + "_flag"] = flag.astype(int)
            total = flag.astype(int).cumsum()
            out[label + "_streak"] = (total - total.where(~flag).ffill().fillna(0)).clip(upper=params.streak_len)
            out[label + "_prior_frequency"] = flag.astype(float).rolling(
                params.baseline_bars, min_periods=params.baseline_bars
            ).mean().shift(params.streak_len)
            out[label + "_onset"] = (
                out[label + "_streak"].eq(params.streak_len)
                & out[label + "_streak"].shift(1).lt(params.streak_len)
                & out[label + "_prior_frequency"].le(params.drought_fraction)
            )
        for label, window, lag in (
            ("recent", params.streak_len, 0),
            ("prior", params.baseline_bars, params.streak_len),
        ):
            out[label + "_volume"] = out["valid_volume"].rolling(window, min_periods=window).sum().shift(lag)
            out[label + "_signed_volume"] = out["signed_volume"].rolling(window, min_periods=window).sum().shift(lag)
            out[label + "_flow"] = out[label + "_signed_volume"] / out[label + "_volume"].where(out[label + "_volume"] > 0)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        ready = out[["atr", "recent_flow", "prior_flow"]].notna().all(axis=1) & out["atr"].gt(0)
        out["entry"] = np.select([
            ready & out["up_onset"] & out["recent_flow"].ge(params.flow_threshold) & out["prior_flow"].le(0),
            ready & out["down_onset"] & out["recent_flow"].le(-params.flow_threshold) & out["prior_flow"].ge(0),
        ], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PressureParams) -> SignalFrame:
        df = pd.DataFrame({"signal": np.zeros(len(data), dtype=int), "size": 1.0}, index=data.index)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        position, entry_bar = 0, -1
        entry_atr, stop = 0.0, np.nan
        signals = np.zeros(len(data), dtype=int)
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # Freeze the stop around the next-bar opening fill, using decision-bar ATR.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                stopped = (
                    not np.isfinite(closes[i]) or not np.isfinite(stop)
                    or (position == 1 and closes[i] < stop)
                    or (position == -1 and closes[i] > stop)
                )
                if stopped or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
            elif i >= self.warmup_bars(params) - 1:
                if entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]):
                    position = int(entries[i])
                    entry_bar, entry_atr = i + 1, float(atr[i])
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = PressureParams(streak_len=2, baseline_bars=2, atr_bars=1, stop_atr=2.0, hold_bars=3)
    index = pd.date_range("2020-01-01", periods=14, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry": 0}, index=index)
    ind.iloc[4, ind.columns.get_loc("entry")] = 1
    ind.iloc[9, ind.columns.get_loc("entry")] = -1
    data.iloc[5, data.columns.get_loc("open")] = 110.0
    data.iloc[5, data.columns.get_loc("close")] = 110.0
    data.iloc[6, data.columns.get_loc("close")] = 107.0
    data.iloc[10, data.columns.get_loc("open")] = 90.0
    data.iloc[10, data.columns.get_loc("close")] = 90.0
    data.iloc[11, data.columns.get_loc("close")] = 93.0
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, -1, -1, 0, 0]
    data.loc[:, "open"] = 100.0
    data.loc[:, "close"] = 100.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, -1, -1, 0]
    assert (result["size"] > 0).all()
    assert strategy.indicators(data, params)["entry"].eq(0).all()


if __name__ == "__main__":
    _self_check()
