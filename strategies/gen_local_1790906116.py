from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    block: int = 5
    effort_ratio: float = 1.25
    travel_ratio: float = 0.65
    minimum_move_atr: float = 1.0
    center_window: int = 3
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790906116"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(2 * params.block + 1, params.atr_window + 1,
                       params.center_window + 2))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.block, params.center_window, params.atr_window,
               params.hold_bars) < 1:
            raise ValueError("Window lengths and holding period must be positive.")
        if not (params.effort_ratio > 1 and 0 < params.travel_ratio < 1
                and params.minimum_move_atr > 0 and params.stop_atr > 0):
            raise ValueError("Invalid exhaustion or stop thresholds.")
        out = pd.DataFrame(index=data.index)
        out["change"] = data["close"].diff()
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["net_travel"] = out["change"].rolling(
            params.block, min_periods=params.block).sum()
        for name, direction in (("down", -1), ("up", 1)):
            out[name + "_volume"] = data["volume"].where(
                direction * out["change"] > 0, 0.0).where(out["change"].notna())
            out[name + "_travel"] = (direction * out["change"]).clip(lower=0)
            for measure in ("volume", "travel"):
                column = name + "_" + measure
                out[column + "_recent"] = out[column].rolling(
                    params.block, min_periods=params.block).sum()
                out[column + "_prior"] = out[column + "_recent"].shift(params.block)
            out[name + "_exhaustion"] = (
                (out[name + "_volume_prior"] > 0)
                & (out[name + "_volume_recent"]
                   >= params.effort_ratio * out[name + "_volume_prior"])
                & (out[name + "_travel_prior"] >= params.minimum_move_atr * out["atr"])
                & (out[name + "_travel_recent"] > 0)
                & (out[name + "_travel_recent"]
                   <= params.travel_ratio * out[name + "_travel_prior"])
                & (direction * out["net_travel"] > 0)
            ).fillna(False)
        out["center"] = (data["high"] + data["low"]) / 2.0
        out["center_change"] = out["center"].diff()
        out["center_drift"] = out["center_change"].rolling(
            params.center_window, min_periods=params.center_window).mean()
        out["previous_drift"] = out["center_drift"].shift(1)
        out["long_setup"] = (
            out["down_exhaustion"] & (out["center_drift"] > 0)
            & (out["previous_drift"] <= 0) & (out["atr"] > 0)
        ).fillna(False)
        out["short_setup"] = (
            out["up_exhaustion"] & (out["center_drift"] < 0)
            & (out["previous_drift"] >= 0) & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        targets = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_setup"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_setup"].fillna(False).to_numpy(dtype=bool)
        side = 0
        entry_price = np.nan
        frozen_atr = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if side:
                # The preceding close's order fills at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                stopped = (
                    not np.isfinite(entry_price) or not np.isfinite(closes[i])
                    or side * (closes[i] - entry_price) <= -params.stop_atr * frozen_atr
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    side = 0
                    targets[i] = 0
                    continue
            elif np.isfinite(atr[i]) and atr[i] > 0:
                side = 1 if longs[i] else (-1 if shorts[i] else 0)
                if side:
                    frozen_atr = atr[i]
                    entry_bar = i + 1
                    entry_price = np.nan
            targets[i] = side
        df["signal"] = targets
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=4)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for side in (1, -1):
        data = pd.DataFrame({
            "open": [100.0] * 4,
            "high": [104.0] * 4,
            "low": [96.0] * 4,
            "close": [100.0, 100.0, 100.0 - side * 3.0, 100.0],
            "volume": [1000.0] * 4,
        }, index=index)
        indicators = pd.DataFrame({
            "atr": [1.0] * 4,
            "long_setup": [side == 1, False, False, False],
            "short_setup": [side == -1, False, False, False],
        }, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, 0]
        data["close"] = 100.0
        result = strategy.generate_signals(
            data, indicators, ctx, StrategyParams(hold_bars=2)).data
        assert result["signal"].tolist() == [0, side, side, 0]
    empty = data.iloc[:0]
    result = strategy.generate_signals(
        empty, strategy.indicators(empty, params), ctx, params).data
    assert result.empty


if __name__ == "__main__":
    _self_check()
