from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    streak_length: int = 3
    flow_window: int = 5
    flow_threshold: float = 0.15
    vol_window: int = 20
    target_vol: float = 0.10
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1790983537"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.vol_window + 1, params.flow_window, params.streak_length + 1))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["relative_range"] = out["range"] / data["close"].where(data["close"] > 0)
        out["shrinking"] = out["relative_range"].lt(out["relative_range"].shift(1)).fillna(False)
        groups = (~out["shrinking"]).cumsum()
        out["streak"] = out["shrinking"].astype(int).groupby(groups).cumsum()
        out["close_location"] = ((2.0 * data["close"] - data["high"] - data["low"]) / out["range"]).clip(-1.0, 1.0).fillna(0.0)
        out["valid_volume"] = data["volume"].clip(lower=0.0).fillna(0.0)
        out["signed_volume"] = out["close_location"] * out["valid_volume"]
        out["flow_sum"] = out["signed_volume"].rolling(params.flow_window, min_periods=params.flow_window).sum()
        out["volume_sum"] = out["valid_volume"].rolling(params.flow_window, min_periods=params.flow_window).sum()
        out["flow"] = (out["flow_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["annual_vol"] = out["return"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["target_size"] = (params.target_vol / out["annual_vol"].where(out["annual_vol"] > 0)).replace([np.inf, -np.inf], np.nan).clip(0.05, 0.95).fillna(0.05)
        out["entry"] = ((out["streak"] >= params.streak_length) & (out["flow"] > params.flow_threshold) & out["annual_vol"].gt(0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.05, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        targets = indicators["target_size"].to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        held_size = 0.05
        age = 0
        first = self.warmup_bars(params) - 1
        for i in range(n):
            if active:
                if age == 0:
                    # The preceding close's entry signal fills at this bar's open.
                    entry_price = opens[i]
                age += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                reached_target = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                # Close-based exits execute at the following open.
                if not valid_entry or reached_target or age >= params.max_hold:
                    active = False
                    age = 0
                    continue
            elif i >= first and entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                age = 0
                held_size = float(np.clip(targets[i], 0.05, 0.95))
            if active:
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = CompressionParams(streak_length=1, flow_window=2, vol_window=2, max_hold=5)
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": False, "target_size": 0.5}, index=index)
    indicators.iloc[2, 0] = True
    timed = strategy.generate_signals(data, indicators, None, params).data
    assert timed["signal"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    data.loc[index[4], "close"] = 103.0
    profited = strategy.generate_signals(data, indicators, None, params).data
    assert profited["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    assert profited["size"].gt(0).all()
    computed = strategy.indicators(data, params)
    assert computed.index.equals(data.index)
    assert np.isfinite(computed["target_size"]).all()
    assert not computed["entry"].iloc[:2].any()


if __name__ == "__main__":
    _self_check()
