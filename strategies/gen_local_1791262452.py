from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    excursion_window: int = 20
    rank_window: int = 60
    atr_window: int = 14
    rank_threshold: float = 0.85
    stop_atr: float = 1.5
    holding_bars: int = 2
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791262452"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if min(params.excursion_window, params.rank_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.rank_threshold <= 1.0:
            raise ValueError("rank_threshold must be in (0, 1]")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        if not np.isfinite(params.size) or params.size <= 0:
            raise ValueError("size must be positive and finite")
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be one or two")
        return int(max(200, params.excursion_window + params.rank_window + 1,
                       params.atr_window + params.rank_window + 2))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["change"] = close.diff()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().where(lambda x: x > 0)
        out["peak"] = close.rolling(params.excursion_window).max()
        out["trough"] = close.rolling(params.excursion_window).min()
        out["volume_mean"] = data["volume"].rolling(params.excursion_window).mean()
        out["relative_volume"] = (
            data["volume"] / out["volume_mean"].where(out["volume_mean"] > 0)
        ).where(data["volume"] > 0)
        out["long_depth"] = (out["peak"] - close) / out["atr"]
        out["short_depth"] = (close - out["trough"]) / out["atr"]
        out["long_gap"] = out["peak"].shift(1) - out["previous_close"]
        out["short_gap"] = out["previous_close"] - out["trough"].shift(1)
        out["previous_atr"] = out["atr"].shift(1)
        out["long_repair"] = (
            out["change"] / out["long_gap"].clip(lower=out["previous_atr"])
            / out["relative_volume"]
        )
        out["short_repair"] = (
            -out["change"] / out["short_gap"].clip(lower=out["previous_atr"])
            / out["relative_volume"]
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        for name in ("long_depth", "short_depth", "long_repair", "short_repair"):
            out[name + "_rank"] = out[name].rolling(
                params.rank_window, min_periods=params.rank_window
            ).rank(pct=True)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        close = data["close"]
        threshold = params.rank_threshold
        long_entry = (
            (close > indicators["ma200"])
            & (indicators["long_depth"].shift(1) >= 1.0)
            & (indicators["long_depth_rank"].shift(1) >= threshold)
            & (indicators["long_repair"] > 0)
            & (indicators["long_repair_rank"] >= threshold)
        )
        short_entry = (
            (close < indicators["ma200"])
            & (indicators["short_depth"].shift(1) >= 1.0)
            & (indicators["short_depth_rank"].shift(1) >= threshold)
            & (indicators["short_repair"] > 0)
            & (indicators["short_repair_rank"] >= threshold)
        )
        ready = (
            np.isfinite(close) & (close > 0)
            & np.isfinite(indicators["atr"]) & (indicators["atr"] > 0)
            & np.isfinite(indicators["ma200"])
        )
        candidates = np.select(
            [(long_entry & ready).fillna(False), (short_entry & ready).fillna(False)],
            [1, -1], default=0,
        )
        closes = close.to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        decision_bar = -1
        entry_atr = 0.0
        stop = np.nan
        for i in range(len(data)):
            if position:
                if i == decision_bar + 1:
                    # Only the current bar's open is used; the ATR is frozen at the decision close.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    stop = opens[i] - position * params.stop_atr * entry_atr
                stopped = (
                    not np.isfinite(closes[i]) or closes[i] <= 0
                    or (position == 1 and closes[i] < stop)
                    or (position == -1 and closes[i] > stop)
                )
                expired = i - decision_bar >= params.holding_bars
                if stopped or expired:
                    position = 0
                raw[i] = position
                continue
            if i >= warmup - 1 and candidates[i] != 0:
                position = int(candidates[i])
                decision_bar = i
                entry_atr = atr[i]
                raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = float(params.size)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=205, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({
        "ma200": 99.0, "atr": 1.0,
        "long_depth": 2.0, "short_depth": 2.0,
        "long_depth_rank": 0.95, "short_depth_rank": 0.95,
        "long_repair": 0.0, "short_repair": 0.0,
        "long_repair_rank": 0.0, "short_repair_rank": 0.0,
    }, index=index)
    ind.loc[index[200], ["long_repair", "short_repair"]] = 1.0
    ind.loc[index[200], ["long_repair_rank", "short_repair_rank"]] = 0.95
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    def signals():
        return strategy.generate_signals(data, ind, ctx, params).data["signal"]
    assert signals().iloc[200:205].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[201], "close"] = 97.0
    ind.loc[index[201], "atr"] = 100.0
    assert signals().iloc[200:204].tolist() == [0, 1, 0, 0]
    data.loc[index[201], ["open", "close"]] = [90.0, 91.0]
    assert signals().iloc[200:205].tolist() == [0, 1, 1, 0, 0]
    ind["ma200"] = 101.0
    data.loc[index[201], ["open", "close"]] = [100.0, 103.0]
    assert signals().iloc[200:204].tolist() == [0, -1, 0, 0]
    actual = strategy.indicators(data, params)
    result = strategy.generate_signals(data, actual, ctx, params).data
    assert result.index.equals(data.index)
    assert result["signal"].isin([-1, 0, 1]).all()
    assert (result["size"] > 0).all()
    assert not result["signal"].iloc[:200].any()


if __name__ == "__main__":
    _self_check()
