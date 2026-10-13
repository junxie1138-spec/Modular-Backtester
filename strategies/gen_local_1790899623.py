from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorbedMomentumParams:
    ma_window: int = 20
    entry_z: float = 0.6
    max_z: float = 2.5
    reset_z: float = 0.25
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[AbsorbedMomentumParams]):
    strategy_id = "gen_local_1790899623"

    @classmethod
    def params_type(cls):
        return AbsorbedMomentumParams

    def warmup_bars(self, params: AbsorbedMomentumParams) -> int:
        return int(params.ma_window + 2)

    def indicators(self, data: pd.DataFrame, params: AbsorbedMomentumParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.max_hold_bars < 1:
            raise ValueError("Lookback must be >= 2 and holding period >= 1.")
        if not (0 <= params.reset_z < params.entry_z < params.max_z):
            raise ValueError("Require 0 <= reset_z < entry_z < max_z.")
        if not (0 < params.profit_target < 1):
            raise ValueError("Profit target must lie between 0 and 1.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["dispersion"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["dispersion"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["price_step"] = close.diff()
        out["ma_step"] = out["ma"].diff()
        out["distance_step"] = out["z"].abs().diff()
        magnitude = out["z"].abs()
        valid = out[["z", "price_step", "ma_step", "distance_step"]].notna().all(axis=1)
        eligible = valid & magnitude.between(params.entry_z, params.max_z) & (out["distance_step"] < 0)
        out["long_candidate"] = eligible & (out["z"] > 0) & (out["price_step"] > 0) & (out["ma_step"] > 0)
        out["short_candidate"] = eligible & (out["z"] < 0) & (out["price_step"] < 0) & (out["ma_step"] < 0)
        out["long_confirmed"] = out["long_candidate"] & out["long_candidate"].shift(1, fill_value=False)
        out["short_confirmed"] = out["short_candidate"] & out["short_candidate"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AbsorbedMomentumParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        longs = indicators["long_confirmed"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_confirmed"].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        armed = True
        for i in range(len(data)):
            if np.isfinite(z[i]) and abs(z[i]) <= params.reset_z:
                armed = True
            if position:
                # Read the entry open only once its bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                        position = 0
                decisions[i] = position
                continue
            if armed and np.isfinite(closes[i]) and closes[i] > 0:
                position = 1 if longs[i] else (-1 if shorts[i] else 0)
                if position:
                    entry_bar = i + 1
                    entry_price = np.nan
                    armed = False
            decisions[i] = position
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (-1, 1):
        for target in (0.02, 0.5):
            data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0}, index=index)
            data.loc[index[3], "close"] = 100.0 + direction * 3.0
            candidate = pd.Series([False, True, True, False, False, False, False], index=index)
            confirmed = candidate & candidate.shift(1, fill_value=False)
            ind = pd.DataFrame({"z": direction * 0.8, "long_confirmed": confirmed if direction == 1 else False, "short_confirmed": confirmed if direction == -1 else False}, index=index)
            params = AbsorbedMomentumParams(profit_target=target, max_hold_bars=3)
            actual = strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist()
            expected = [0, 0, 0, direction, 0, 0, 0] if target == 0.02 else [0, 0, 0, direction, direction, direction, 0]
            assert actual == expected
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=40))
    params = AbsorbedMomentumParams()
    ind = strategy.indicators(flat, params)
    assert not ind[["long_confirmed", "short_confirmed"]].to_numpy().any()
    assert strategy.generate_signals(flat, ind, ctx, params).data["signal"].eq(0).all()
