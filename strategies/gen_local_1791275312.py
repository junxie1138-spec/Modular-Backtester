from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExcursionParams:
    ma_window: int = 40
    regime_window: int = 20
    growth_lag: int = 5
    atr_window: int = 14
    entry_z: float = 1.1
    growth_threshold: float = 0.10
    saturation: float = 0.65
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ExcursionParams]):
    strategy_id = "gen_local_1791275312"

    @classmethod
    def params_type(cls) -> type[ExcursionParams]:
        return ExcursionParams

    def warmup_bars(self, params: ExcursionParams) -> int:
        return int(max(params.ma_window + params.regime_window + params.growth_lag + 2,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ExcursionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["infected"] = out["z"].abs().ge(1.0).astype(float).where(out["z"].notna())
        out["occupancy"] = out["infected"].rolling(
            params.regime_window, min_periods=params.regime_window).mean()
        out["growth"] = out["occupancy"] - out["occupancy"].shift(params.growth_lag)
        out["spreading"] = (out["growth"].ge(params.growth_threshold)
                            & out["occupancy"].lt(params.saturation))
        out["receding"] = (out["occupancy"].ge(params.saturation)
                           & out["growth"].lt(0.0))
        out["candidate"] = np.select([
            (out["spreading"] & out["z"].ge(params.entry_z))
            | (out["receding"] & out["z"].le(-params.entry_z)),
            (out["spreading"] & out["z"].le(-params.entry_z))
            | (out["receding"] & out["z"].ge(params.entry_z))
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ExcursionParams
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = previous_candidate = streak = 0
        entry_bar = -1
        entry_atr = stop_price = np.nan
        for i in range(len(data)):
            if position:
                # Entry fills on the bar after confirmation; freeze its ATR distance.
                if i == entry_bar:
                    stop_price = opens[i] - position * params.stop_atr * entry_atr
                stopped = (not np.isfinite(close[i]) or not np.isfinite(stop_price)
                           or position * (close[i] - stop_price) < 0.0)
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
                previous_candidate = streak = 0
            else:
                candidate = int(candidates[i])
                valid = (np.isfinite(close[i]) and close[i] > 0.0
                         and np.isfinite(atr[i]) and atr[i] > 0.0)
                if not valid:
                    candidate = 0
                streak = streak + 1 if candidate and candidate == previous_candidate else int(candidate != 0)
                previous_candidate = candidate
                if streak >= 2:
                    position = candidate
                    entry_bar = i + 1
                    entry_atr = atr[i]
                    stop_price = np.nan
                    previous_candidate = streak = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = ExcursionParams()
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0
    }, index=index)
    for direction in (1, -1):
        bars = data.copy()
        entry = 100.0 + direction * 10.0
        bars.loc[index[2], "open"] = entry
        bars.loc[index[2]:, "close"] = entry
        bars.loc[index[3], "close"] = entry - direction * 4.0
        bars.loc[index[4], "close"] = entry - direction * 6.0
        features = pd.DataFrame({"atr": 2.0, "candidate": 0}, index=index)
        features.loc[index[:2], "candidate"] = direction
        features.loc[index[3]:, "atr"] = 100.0
        result = GeneratedStrategy.generate_signals(None, bars, features, None, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction] + [0] * 19
        assert result["size"].gt(0).all()
    features = pd.DataFrame({"atr": 2.0, "candidate": 0}, index=index)
    features.loc[index[:2], "candidate"] = 1
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].tolist() == [0, 0] + [1] * 18 + [0] * 4
    features.loc[index[1], "candidate"] = 0
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].eq(0).all()
    flat = GeneratedStrategy.indicators(None, data, params)
    assert flat["candidate"].eq(0).all()


if __name__ == "__main__":
    _self_check()
