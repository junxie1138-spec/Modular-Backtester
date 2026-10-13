from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuarterLaunchParams:
    breakout_days: int = 10
    volume_days: int = 20
    volume_ratio: float = 1.2
    atr_days: int = 14
    stop_atr: float = 1.8
    holding_days: int = 4


class GeneratedStrategy(BaseStrategy[QuarterLaunchParams]):
    strategy_id = "gen_local_1791231123"

    @classmethod
    def params_type(cls) -> type[QuarterLaunchParams]:
        return QuarterLaunchParams

    def warmup_bars(self, params: QuarterLaunchParams) -> int:
        return max(200, params.breakout_days + 1, params.volume_days + 1, params.atr_days + 1)

    def indicators(self, data: pd.DataFrame, params: QuarterLaunchParams) -> pd.DataFrame:
        if min(params.breakout_days, params.volume_days, params.atr_days) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 3 <= params.holding_days <= 5:
            raise ValueError("holding_days must be between 3 and 5.")
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0
                and np.isfinite(params.volume_ratio) and params.volume_ratio > 0):
            raise ValueError("Stop and volume multipliers must be positive and finite.")
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["atr"] = true_range.rolling(params.atr_days, min_periods=params.atr_days).mean()
        out["breakout_level"] = close.shift(1).rolling(
            params.breakout_days, min_periods=params.breakout_days
        ).max()
        out["volume_base"] = data["volume"].shift(1).rolling(
            params.volume_days, min_periods=params.volume_days
        ).mean()
        out["volume_ratio"] = data["volume"].div(out["volume_base"].where(out["volume_base"] > 0))
        out["close_location"] = (close - data["low"]).div(
            (data["high"] - data["low"]).where(data["high"] > data["low"])
        )
        out["quarter_window"] = (
            data.index.month.isin([1, 4, 7, 10]) & (data.index.day <= 7)
        )
        out["entry"] = (
            out["quarter_window"]
            & (close > out["ma200"])
            & (close > out["breakout_level"])
            & (close > data["open"])
            & (out["volume_ratio"] >= params.volume_ratio)
            & (out["close_location"] >= 0.75)
            & (out["atr"] > 0)
            & np.isfinite(out[["ma200", "atr", "breakout_level",
                               "volume_ratio", "close_location"]]).all(axis=1)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: QuarterLaunchParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        pending = False
        stop = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    entry_bar = i
                    # Fix the stop to the entry open and the decision bar's ATR.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
            if entry_bar >= 0:
                if (not np.isfinite(close[i]) or close[i] < stop
                        or i - entry_bar + 1 >= params.holding_days):
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                pending = True
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = QuarterLaunchParams(stop_atr=2.0)
    index = pd.date_range("2024-01-01", periods=7, freq="B")
    data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "entry": [True, False, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data.loc[index[1], "open"] = 105.0
    data.loc[index[1:3], "close"] = [104.0, 102.0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
