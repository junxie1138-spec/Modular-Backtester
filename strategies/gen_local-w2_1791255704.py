from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpiryDiscountParams:
    ma_bars: int = 20
    percentile_bars: int = 252
    entry_percentile: float = 0.35
    atr_bars: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8
    pre_expiry_days: int = 3


class GeneratedStrategy(BaseStrategy[ExpiryDiscountParams]):
    strategy_id = "gen_local-w2_1791255704"

    @classmethod
    def params_type(cls) -> type[ExpiryDiscountParams]:
        return ExpiryDiscountParams

    def warmup_bars(self, params: ExpiryDiscountParams) -> int:
        return int(max(params.ma_bars + params.percentile_bars, params.ma_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: ExpiryDiscountParams) -> pd.DataFrame:
        if min(params.ma_bars, params.percentile_bars, params.atr_bars, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0 or params.pre_expiry_days < 0:
            raise ValueError("Invalid stop multiplier or expiry window")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out["price_std"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["price_std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["z_change"] = out["z"].diff()
        out["cutoff"] = out["z"].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.entry_percentile)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        index = data.index
        first_weekday = (index.dayofweek - (index.day - 1)) % 7
        out["expiry_day"] = 15 + (4 - first_weekday) % 7
        out["days_to_expiry"] = out["expiry_day"] - index.day
        out["cycle"] = index.year * 12 + index.month
        out["eligible"] = (
            out["days_to_expiry"].between(0, params.pre_expiry_days)
            & (out["z"] <= out["cutoff"])
            & (out["z_change"] > 0)
            & np.isfinite(out["atr"])
            & (out["atr"] > 0)
            & np.isfinite(close)
            & (close > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ExpiryDiscountParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        cycles = indicators["cycle"].to_numpy()
        signal = np.zeros(len(data), dtype=int)
        active = pending = False
        frozen_atr = stop = 0.0
        age = 0
        last_cycle = -1
        for i in range(len(data)):
            if pending:
                pending = False
                active = bool(np.isfinite(opening[i]) and opening[i] > 0)
                if active:
                    # Entry occurs at this open; ATR was frozen at the preceding decision close.
                    stop = opening[i] - params.stop_atr * frozen_atr
                    age = 0
            if active:
                age += 1
                if not np.isfinite(close[i]) or close[i] < stop or age >= params.holding_bars:
                    active = False
                else:
                    signal[i] = 1
                continue
            if eligible[i] and cycles[i] != last_cycle:
                pending = True
                frozen_atr = atr[i]
                last_cycle = cycles[i]
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2026-02-17", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [104.0] * 5,
        "low": [96.0] * 5,
        "close": [100.0, 99.0, 97.0, 102.0, 103.0],
        "volume": [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 10.0, 10.0, 10.0, 10.0],
        "eligible": [True] * 5,
        "cycle": [2026 * 12 + 2] * 5,
    }, index=index)
    strategy = GeneratedStrategy()
    params = ExpiryDiscountParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    data["close"] = 100.0
    params.holding_bars = 2
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert strategy.indicators(data, params)["expiry_day"].tolist() == [20] * 5


if __name__ == "__main__":
    _self_check()
