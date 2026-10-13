from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 20
    percentile_window: int = 126
    breakout_percentile: float = 0.85
    noise_window: int = 3
    profit_target: float = 0.008
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791323673"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.ma_window, params.noise_window) + params.percentile_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.ma_window < 2 or params.noise_window < 1 or params.percentile_window < 2:
            raise ValueError("Indicator windows must be positive; MA and percentile windows must exceed one.")
        if not 0.5 < params.breakout_percentile < 1.0:
            raise ValueError("breakout_percentile must be between 0.5 and 1.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["price_std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["z_threshold"] = out["z"].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.breakout_percentile)
        out["daily_range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["wick_noise"] = ((out["daily_range"] - (close - data["open"]).abs()) / out["daily_range"]).clip(0.0, 1.0)
        out["noise"] = out["wick_noise"].rolling(params.noise_window, min_periods=params.noise_window).mean()
        out["noise_threshold"] = out["noise"].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(0.5)
        out["previous_high"] = data["high"].shift(1)
        out["entry"] = (
            (out["z"] > out["z_threshold"])
            & (out["z"].shift(1) <= out["z_threshold"].shift(1))
            & (close > out["ma"])
            & (close > out["previous_high"])
            & (out["noise"] < out["noise_threshold"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.hold_bars not in (1, 2) or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Use a positive profit target and a one- or two-bar holding period.")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_price = 0.0
        held = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    held = 0
                    active = True
            if active:
                held += 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or held >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signals[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0, "close": [100.0, 100.0, 102.0, 100.0, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, True, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params(profit_target=0.01)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
