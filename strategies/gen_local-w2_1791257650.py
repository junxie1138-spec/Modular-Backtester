from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangePulseParams:
    edge_threshold: float = 0.12
    stop_atr: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.edge_threshold) or self.edge_threshold <= 0:
            raise ValueError("edge_threshold must be positive and finite")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")


class GeneratedStrategy(BaseStrategy[RangePulseParams]):
    strategy_id = "gen_local-w2_1791257650"

    @classmethod
    def params_type(cls):
        return RangePulseParams

    def warmup_bars(self, params: RangePulseParams) -> int:
        return 272

    def indicators(self, data: pd.DataFrame, params: RangePulseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["pressure"] = (
            (2 * data["close"] - data["high"] - data["low"]) / out["range"]
        ).clip(-1, 1)
        out["range_change"] = np.log(out["range"] / out["range"].shift(1))
        out["change_lag1"] = out["range_change"].shift(1)
        out["change_lag2"] = out["range_change"].shift(2)
        out["range_level"] = np.log(
            out["range"] / out["range"].rolling(10, min_periods=10).mean()
        )
        # At bar m, this outcome belongs to the analog ending at m-2.
        out["matured_response"] = (
            (data["close"] - data["open"].shift(1))
            / out["atr"].shift(2)
            * np.sign(out["pressure"].shift(2))
        )
        features = out[
            ["range_change", "change_lag1", "change_lag2", "range_level"]
        ].to_numpy(dtype=float)
        response = out["matured_response"].to_numpy(dtype=float)
        pressure = out["pressure"].to_numpy(dtype=float)
        forecast = np.zeros(len(data), dtype=float)
        # ponytail: bounded 252-bar scan; index neighbors only if history grows.
        for i in range(272, len(data)):
            if not np.isfinite(features[i]).all() or not np.isfinite(pressure[i]):
                continue
            if abs(pressure[i]) < 0.2:
                continue
            candidates = np.arange(max(14, i - 252), i - 5)
            valid = (
                np.isfinite(features[candidates]).all(axis=1)
                & np.isfinite(response[candidates + 2])
            )
            candidates = candidates[valid]
            if len(candidates) < 12:
                continue
            distance = np.sum((features[candidates] - features[i]) ** 2, axis=1)
            nearest = np.argsort(distance, kind="stable")[:12]
            weights = 1.0 / (0.05 + distance[nearest])
            forecast[i] = np.sign(pressure[i]) * np.average(
                response[candidates[nearest] + 2], weights=weights
            )
        out["forecast"] = forecast
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangePulseParams
    ) -> SignalFrame:
        count = len(data)
        decisions = np.zeros(count, dtype=int)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        position = 0
        pending = False
        age = 0
        entry_atr = 0.0
        stop = 0.0
        for i in range(count):
            if pending:
                pending = False
                if not np.isfinite(opening[i]) or opening[i] <= 0:
                    position = 0
                    continue
                # The prior close's decision fills at this bar's open.
                stop = opening[i] - position * params.stop_atr * entry_atr
                age = 0
            if position:
                age += 1
                stopped = (
                    not np.isfinite(closing[i])
                    or position * (closing[i] - stop) < 0
                )
                if stopped or age >= 2:
                    position = 0
                    continue
            elif (
                np.isfinite(closing[i]) and closing[i] > 0
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(forecast[i])
                and abs(forecast[i]) >= params.edge_threshold
            ):
                position = 1 if forecast[i] > 0 else -1
                entry_atr = atr[i]
                pending = True
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    indicators = pd.DataFrame(
        {"atr": 1.0, "forecast": [1, 0, 0, -1, 0, 0, 0, 0]}, index=index
    )
    params = RangePulseParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 0]
    data.loc[index[1], "close"] = 98.0
    data.loc[index[4], "close"] = 102.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
