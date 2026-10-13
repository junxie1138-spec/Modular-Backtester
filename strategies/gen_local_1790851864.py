from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RejectionParams:
    rank_window: int = 40
    volume_window: int = 20
    center_window: int = 5
    atr_window: int = 14
    entry_percentile: float = 0.80
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[RejectionParams]):
    strategy_id = "gen_local_1790851864"

    @classmethod
    def params_type(cls):
        return RejectionParams

    def warmup_bars(self, params: RejectionParams) -> int:
        return int(max(params.volume_window + params.rank_window + 1,
                       params.center_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RejectionParams) -> pd.DataFrame:
        if min(params.rank_window, params.volume_window,
               params.center_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if not (0.5 < params.entry_percentile <= 1.0
                and np.isfinite(params.breakeven_pct) and params.breakeven_pct > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0
                and np.isfinite(params.initial_stop_atr) and params.initial_stop_atr > 0
                and 3 <= params.max_hold <= 5):
            raise ValueError("Invalid entry, stop, or holding parameters.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["lower_wick"] = (
            data[["open", "close"]].min(axis=1) - data["low"]
        ).clip(lower=0.0)
        out["mean_volume"] = data["volume"].rolling(params.volume_window).mean()
        out["rejection"] = (
            out["lower_wick"] / out["range"]
            * data["volume"] / out["mean_volume"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out["rejection_rank"] = out["rejection"].rolling(params.rank_window).rank(pct=True)
        out["center"] = data["close"].rolling(params.center_window).mean()
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["setup"] = (
            out["rejection_rank"].ge(params.entry_percentile)
            & data["close"].lt(out["center"])
            & out["atr"].gt(0.0)
        ).fillna(False)
        out["confirmed"] = (
            out["setup"] & out["setup"].shift(1, fill_value=False)
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RejectionParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                entry = float(data["open"].iat[i])
                previous_atr = float(indicators["atr"].iat[i - 1])
                pending = False
                if np.isfinite(entry) and entry > 0 and np.isfinite(previous_atr) and previous_atr > 0:
                    active = True
                    armed = False
                    entry_bar = i
                    peak = entry
                    stop = entry - params.initial_stop_atr * previous_atr

            if active:
                high = float(data["high"].iat[i])
                low = float(data["low"].iat[i])
                atr = float(indicators["atr"].iat[i])
                # Test the previously established stop before ratcheting it.
                # Every exit is decided at this close and filled next open.
                if (not np.isfinite(high) or not np.isfinite(low)
                        or not np.isfinite(atr) or atr <= 0
                        or low <= stop or i - entry_bar + 1 >= params.max_hold):
                    active = False
                    continue
                peak = max(peak, high)
                if high >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr)
                decisions[i] = 1
            elif bool(indicators["confirmed"].iat[i]):
                pending = True
                decisions[i] = 1

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    close = np.array([100, 99, 98, 97, 96, 96, 97, 97, 97, 97, 97, 97], dtype=float)
    data = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.1,
        "close": close, "volume": 1000.0,
    }, index=index)
    data.loc[index[3], "low"] = 94.0
    data.loc[index[4], "low"] = 92.0
    params = RejectionParams(rank_window=2, volume_window=2, center_window=3, atr_window=2)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, ind, ctx, params).data
    assert ind["setup"].iat[3] and not ind["confirmed"].iat[3]
    assert ind["confirmed"].iat[4]
    assert signals["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[7], "low"] = 95.9
    stopped = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert stopped["signal"].iat[7] == 1 and stopped["signal"].iat[8] == 0
    prefix = data.iloc[:8]
    prefix_signals = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(stopped.iloc[:8], prefix_signals)


if __name__ == "__main__":
    _self_check()
