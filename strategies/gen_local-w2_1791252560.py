from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RetestParams:
    ma_window: int = 20
    percentile_window: int = 126
    tail_fraction: float = 0.20
    retest_lag: int = 3
    atr_window: int = 14
    stop_atr: float = 1.2
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[RetestParams]):
    strategy_id = "gen_local-w2_1791252560"

    @classmethod
    def params_type(cls) -> type[RetestParams]:
        return RetestParams

    def warmup_bars(self, params: RetestParams) -> int:
        return int(max(params.ma_window + params.percentile_window + params.retest_lag + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RetestParams) -> pd.DataFrame:
        if (params.ma_window < 2 or params.percentile_window < 2
                or params.retest_lag < 3 or params.atr_window < 1
                or not 0.0 < params.tail_fraction < 0.5
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0
                or params.holding_bars not in (1, 2)):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0).replace(0.0, np.nan)
        out["z"] = ((close - out["ma"]) / out["std"]).replace([np.inf, -np.inf], np.nan)
        historical_z = out["z"].shift(1)
        out["lower_tail"] = historical_z.rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.tail_fraction)
        out["upper_tail"] = historical_z.rolling(params.percentile_window, min_periods=params.percentile_window).quantile(1.0 - params.tail_fraction)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out.loc[out["previous_close"].isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        lag = params.retest_lag
        out["old_close"] = close.shift(lag)
        out["old_left_close"] = close.shift(lag + 1)
        out["old_right_close"] = close.shift(lag - 1)
        out["old_z"] = out["z"].shift(lag)
        out["recent_z"] = out["z"].shift(1)
        out["old_lower_tail"] = out["lower_tail"].shift(lag)
        out["old_upper_tail"] = out["upper_tail"].shift(lag)
        out["recent_lower_tail"] = out["lower_tail"].shift(1)
        out["recent_upper_tail"] = out["upper_tail"].shift(1)
        out["long_entry"] = (
            (out["old_close"] < out["old_left_close"])
            & (out["old_close"] < out["old_right_close"])
            & (out["old_z"] < out["old_lower_tail"])
            & (out["previous_close"] < out["old_close"])
            & (out["previous_close"] < close.shift(2))
            & (out["recent_z"] > out["old_z"])
            & (out["recent_z"] < out["recent_lower_tail"])
            & (close > out["previous_close"])
            & (out["z"] > out["recent_z"])
        ).fillna(False)
        out["short_entry"] = (
            (out["old_close"] > out["old_left_close"])
            & (out["old_close"] > out["old_right_close"])
            & (out["old_z"] > out["old_upper_tail"])
            & (out["previous_close"] > out["old_close"])
            & (out["previous_close"] > close.shift(2))
            & (out["recent_z"] < out["old_z"])
            & (out["recent_z"] > out["recent_upper_tail"])
            & (close < out["previous_close"])
            & (out["z"] < out["recent_z"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RetestParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=np.int64)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan
        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    entry_price = opens[i]
                invalid = not (np.isfinite(entry_price) and entry_price > 0.0 and np.isfinite(closes[i]))
                stopped = (position == 1 and closes[i] < entry_price - params.stop_atr * entry_atr) or (position == -1 and closes[i] > entry_price + params.stop_atr * entry_atr)
                expired = i - entry_bar + 1 >= params.holding_bars
                if invalid or stopped or expired:
                    position = 0
                signals[i] = position
                continue
            if np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                if longs[i] and not shorts[i]:
                    position = 1
                elif shorts[i] and not longs[i]:
                    position = -1
                if position != 0:
                    entry_bar = i + 1
                    entry_price = np.nan
                    entry_atr = atr[i]
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": [100., 100., 99., 100., 100.], "high": [101., 101., 100., 101., 104.], "low": [99., 97., 98., 99., 99.], "close": [100., 98., 99., 100., 103.], "volume": [1000.] * 5}, index=index)
    ind = pd.DataFrame({"atr": [1.] * 5, "long_entry": [True, False, False, False, False], "short_entry": [False, False, True, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, RetestParams())
    assert result.data["signal"].tolist() == [0, 1, 0, -1, -1]
    assert (result.data["size"] > 0.0).all()
    ind["short_entry"] = False
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, RetestParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
