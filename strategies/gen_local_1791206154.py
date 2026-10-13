from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    flow_window: int = 5
    history_bars: int = 126
    entry_quantile: float = 0.85
    volume_quantile: float = 0.60
    atr_bars: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local_1791206154"

    @classmethod
    def params_type(cls) -> type[GapParams]:
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        return int(params.history_bars + max(params.flow_window, params.atr_bars) + 2)

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        if min(params.flow_window, params.history_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (0 < params.entry_quantile < 1 and 0 < params.volume_quantile < 1):
            raise ValueError("Quantiles must lie strictly between zero and one.")
        if not (np.isfinite(params.trail_atr) and params.trail_atr > 0
                and np.isfinite(params.breakeven_pct) and params.breakeven_pct > 0):
            raise ValueError("Stop parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["previous_close"] = close.shift(1)
        out["gap_return"] = np.log(opening / out["previous_close"])
        out["day_return"] = np.log(close / opening)
        out["gap_mean"] = out["gap_return"].rolling(params.flow_window).mean()
        out["weighted_day_return"] = out["day_return"] * volume
        out["volume_sum"] = volume.rolling(params.flow_window).sum()
        out["intraday_flow"] = (
            out["weighted_day_return"].rolling(params.flow_window).sum()
            / out["volume_sum"].replace(0, np.nan)
        )
        out["rejection_pressure"] = -out["gap_mean"] * out["intraday_flow"]
        out["pressure_threshold"] = (
            out["rejection_pressure"].shift(1)
            .rolling(params.history_bars).quantile(params.entry_quantile)
        )
        out["volume_threshold"] = (
            volume.shift(1).rolling(params.history_bars).quantile(params.volume_quantile)
        )
        out["net_return"] = np.log(close / close.shift(params.flow_window))
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        valid = (
            (out["rejection_pressure"] > 0)
            & (out["rejection_pressure"] > out["pressure_threshold"])
            & (volume > out["volume_threshold"])
            & (out["intraday_flow"] * out["net_return"] > 0)
            & (out["atr"] > 0)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        valid &= out[["rejection_pressure", "pressure_threshold", "intraday_flow",
                      "net_return", "volume_threshold", "atr"]].notna().all(axis=1)
        out["entry_signal"] = np.where(
            valid, np.sign(out["intraday_flow"].fillna(0)), 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_signal"].fillna(0).to_numpy(dtype=int)
        side = 0
        entry = np.nan
        stop = 0.0
        peak = 0.0
        armed = False
        age = 0
        for i in range(len(data)):
            exited = False
            if side:
                if not np.isfinite(entry):
                    # Next-bar open is the entry reference for the prior close's decision.
                    entry = opening[i]
                    if i == 0 or not (np.isfinite(entry) and entry > 0
                                      and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                        side = 0
                        exited = True
                    else:
                        stop = -params.trail_atr * atr[i - 1]
                        peak = 0.0
                        armed = False
                        age = 0
                if side:
                    age += 1
                    if not (np.isfinite(closing[i]) and closing[i] > 0
                            and np.isfinite(atr[i]) and atr[i] > 0):
                        side = 0
                        exited = True
                    else:
                        # Stops are evaluated at the close and exits fill next bar.
                        # Signed profit coordinates ratchet upward for longs and shorts.
                        profit = side * (closing[i] - entry)
                        if profit <= stop or age >= params.hold_bars:
                            side = 0
                            exited = True
                        else:
                            peak = max(peak, profit)
                            armed = armed or peak >= params.breakeven_pct * entry
                            if armed:
                                stop = max(stop, 0.0, peak - params.trail_atr * atr[i])
            if not side and not exited and entries[i] in (-1, 1):
                side = int(entries[i])
                entry = np.nan
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = GapParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=6)
    for side in (1, -1):
        prices = 100.0 + side * np.array([0.0, 1.0, 3.0, 2.0, 0.0, 0.0])
        data = pd.DataFrame({"open": 100.0, "close": prices}, index=index)
        derived = pd.DataFrame(
            {"atr": 1.0, "entry_signal": [side, 0, 0, 0, 0, 0]}, index=index
        )
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, side, 0]
        assert (result["size"] > 0).all()
        derived["entry_signal"] = 0
        assert not strategy.generate_signals(data, derived, None, params).data["signal"].any()
    params = GapParams(hold_bars=3, trail_atr=100.0)
    derived["entry_signal"] = [1, 0, 0, 0, 0, 0]
    data["close"] = 100.0
    assert strategy.generate_signals(data, derived, None, params).data["signal"].tolist() == [0, 1, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
