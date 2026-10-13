from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    rank_window: int = 63
    history: int = 126
    radius: float = 0.18
    min_analogs: int = 6
    min_edge: float = 0.001
    atr_window: int = 14
    stop_k: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790862116"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.rank_window + params.history + 4, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not (
            params.rank_window >= 2 and params.history >= params.min_analogs >= 1
            and 0 < params.radius <= 1 and params.min_edge > 0
            and params.atr_window >= 1 and params.stop_k > 0
            and params.hold_bars in (1, 2)
        ):
            raise ValueError("Invalid analog parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["rank"] = out["return"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["rank_lag1"] = out["rank"].shift(1)
        out["rank_lag2"] = out["rank"].shift(2)
        shape = out[["rank_lag2", "rank_lag1", "rank"]]
        out["analog_count"] = 0
        out["successor_sum"] = 0.0
        # ponytail: O(bars * history); batch nearest-neighbor search if history exceeds 1000.
        for lag in range(2, params.history + 2):
            past = shape.shift(lag)
            distance = (shape - past).abs().max(axis=1)
            successor = out["return"].shift(lag - 1)
            match = (
                shape.notna().all(axis=1) & past.notna().all(axis=1)
                & successor.notna() & (distance <= params.radius)
            )
            out["analog_count"] += match.astype(int)
            out["successor_sum"] += successor.where(match, 0.0)
        out["forecast"] = out["successor_sum"] / out["analog_count"].replace(0, np.nan)
        ready = (
            (np.arange(len(data)) >= self.warmup_bars(params) - 1)
            & (out["analog_count"] >= params.min_analogs)
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
        )
        out["setup"] = np.select([
            ready & (out["forecast"] >= params.min_edge),
            ready & (out["forecast"] <= -params.min_edge),
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0
        setup = indicators["setup"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and np.isfinite(closes[i])
                stopped = valid and (
                    closes[i] < entry_price - params.stop_k * entry_atr
                    if position == 1 else
                    closes[i] > entry_price + params.stop_k * entry_atr
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if not valid or stopped or expired:
                    position = 0
            elif (
                i > 0 and setup[i] != 0 and setup[i] == setup[i - 1]
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(closes[i])
            ):
                position = int(setup[i])
                entry_bar = i + 1
                entry_atr = atr[i]
                entry_price = np.nan
            decisions[i] = position
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AnalogParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 97.0,
        "close": [100, 100, 99, 100, 100, 100, 103, 100, 100],
        "volume": 1000.0,
    }, index=index)
    features = pd.DataFrame({
        "setup": [1, 1, 0, 0, -1, -1, 0, 0, 0],
        "atr": [1, 1, 50, 1, 1, 1, 1, 1, 1],
    }, index=index)
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, -1, 0, 0]
    data.loc[index[2], "close"] = 98.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].iloc[3] == 0
    features["setup"] = [1, 0, -1, 0, 1, 0, -1, 0, 0]
    assert not strategy.generate_signals(data, features, None, params).data["signal"].any()
    x = np.arange(260, dtype=float)
    close = 100 * np.exp(0.0002 * x + 0.02 * np.sin(x / 4))
    market = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=pd.date_range("2021-01-01", periods=len(x)))
    full = strategy.indicators(market, params)
    prefix = strategy.indicators(market.iloc[:220], params)
    pd.testing.assert_frame_equal(full.iloc[:220], prefix)
    assert (full["setup"].iloc[:strategy.warmup_bars(params) - 1] == 0).all()


if __name__ == "__main__":
    _self_check()
