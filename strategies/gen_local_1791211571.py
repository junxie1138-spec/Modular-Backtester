from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAcceptanceParams:
    rank_window: int = 60
    overlap_tail: float = 0.20
    body_percentile: float = 0.80
    profit_target: float = 0.015
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RangeAcceptanceParams]):
    strategy_id = "gen_local_1791211571"

    @classmethod
    def params_type(cls):
        return RangeAcceptanceParams

    def warmup_bars(self, params):
        return int(max(200, params.rank_window + 1))

    def indicators(self, data, params):
        if params.rank_window < 2:
            raise ValueError("rank_window must be at least 2")
        df = pd.DataFrame(index=data.index)
        df["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        df["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        df["previous_high"] = data["high"].shift(1)
        df["previous_low"] = data["low"].shift(1)
        df["previous_range"] = df["range"].shift(1)
        df["overlap"] = (
            np.minimum(data["high"], df["previous_high"])
            - np.maximum(data["low"], df["previous_low"])
        ).clip(lower=0)
        denominator = np.minimum(df["range"], df["previous_range"])
        df["overlap_fraction"] = df["overlap"].div(denominator).clip(0, 1)
        df["body"] = (data["close"] - data["open"]).div(df["range"])
        df["body_abs"] = df["body"].abs()
        df["body_rank"] = df["body_abs"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        df["overlap_rank"] = df["overlap_fraction"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        return df.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        if not (0 < params.overlap_tail < 0.5):
            raise ValueError("overlap_tail must lie between 0 and 0.5")
        if not (0 < params.body_percentile <= 1):
            raise ValueError("body_percentile must lie between 0 and 1")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0 or params.hold_bars < 1:
            raise ValueError("profit_target and hold_bars must be positive")
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        valid = (
            np.isfinite(indicators[["ma200", "body", "body_rank", "overlap_rank"]]).all(axis=1)
            & np.isfinite(data["open"]) & np.isfinite(data["close"])
            & (data["open"] > 0) & (data["close"] > 0)
        )
        trend = pd.Series(np.sign(data["close"] - indicators["ma200"]), index=data.index)
        body_direction = np.sign(indicators["body"])
        elastic = (
            (indicators["overlap_rank"] >= 1 - params.overlap_tail)
            & (body_direction == -trend)
        )
        plastic = (
            (indicators["overlap_rank"] <= params.overlap_tail)
            & (body_direction == trend)
        )
        candidate = trend.where(
            valid & (trend != 0)
            & (indicators["body_rank"] >= params.body_percentile)
            & (elastic | plastic), 0
        ).fillna(0).astype(int).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            if side:
                # Entry is the current open after the preceding close's decision.
                if held == 0:
                    entry_price = opening[i]
                held += 1
                gain = (
                    side * (close[i] / entry_price - 1)
                    if np.isfinite(entry_price) and entry_price > 0 else np.nan
                )
                if (
                    not np.isfinite(gain)
                    or gain >= params.profit_target
                    or held >= params.hold_bars
                ):
                    side = 0
                    held = 0
                    entry_price = np.nan
                raw[i] = side
            elif candidate[i]:
                side = int(candidate[i])
                held = 0
                raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeAcceptanceParams(hold_bars=3)
    index = pd.date_range("2020-01-01", periods=8)
    for side in (1, -1):
        data = pd.DataFrame({
            "open": 100.0, "high": 103.0, "low": 97.0,
            "close": [100.0, 100.0, 100.0 + side * 2, 100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": 1000.0,
        }, index=index)
        indicators = pd.DataFrame({
            "ma200": 100.0 - side, "body": float(side),
            "body_rank": 1.0, "overlap_rank": 0.1,
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, side, side, 0, side, side, side, 0]
        assert (result["size"] > 0).all()
        indicators["overlap_rank"] = 0.9
        indicators["body"] = float(-side)
        assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == result["signal"].tolist()
    short_data = data.iloc[:2]
    warm = strategy.indicators(short_data, params)
    assert strategy.generate_signals(short_data, warm, None, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(RangeAcceptanceParams(rank_window=250)) == 251


if __name__ == "__main__":
    _self_check()
