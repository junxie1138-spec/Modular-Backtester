from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    fd_window: int = 32
    fd_order: float = 0.45
    memory: int = 40
    atr_window: int = 14
    echo_min: float = 0.12
    forecast_min: float = 0.20
    stop_atr: float = 1.5

    def __post_init__(self):
        if not (self.fd_window >= 2 and self.memory >= 3 and self.atr_window >= 1):
            raise ValueError("Lookbacks are too short")
        if not (0.0 < self.fd_order < 1.0 and 0.0 < self.echo_min < 1.0):
            raise ValueError("Invalid fractional order or echo threshold")
        if not (np.isfinite(self.forecast_min) and self.forecast_min > 0.0
                and np.isfinite(self.stop_atr) and self.stop_atr > 0.0):
            raise ValueError("Forecast and stop thresholds must be finite and positive")


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local-w2_1791227898"

    @classmethod
    def params_type(cls):
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(200, params.fd_window + 2 * params.memory + 3,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        out = pd.DataFrame(index=data.index)
        fd = close.copy()
        weight = 1.0
        # ponytail: fixed-width fractional filter; increase fd_window for longer memory.
        for lag in range(1, params.fd_window):
            weight *= (lag - 1.0 - params.fd_order) / lag
            fd = fd + weight * close.shift(lag)
        out["fd_close"] = fd
        out["increment"] = fd.diff()
        history = out["increment"].rolling(params.memory, min_periods=params.memory)
        out["increment_mean"] = history.mean().shift(1)
        out["increment_std"] = history.std(ddof=0).shift(1).replace(0.0, np.nan)
        out["z"] = (out["increment"] - out["increment_mean"]) / out["increment_std"]
        window = out["z"].rolling(params.memory, min_periods=params.memory)
        out["rho1"] = window.corr(out["z"].shift(1)).clip(-1.0, 1.0)
        out["rho2"] = window.corr(out["z"].shift(2)).clip(-1.0, 1.0)
        out["echo_score"] = out["rho2"] * out["z"].shift(1)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        score = indicators["echo_score"].to_numpy(dtype=float)
        rho1 = indicators["rho1"].to_numpy(dtype=float)
        rho2 = indicators["rho2"].to_numpy(dtype=float)
        valid = (np.isfinite(close) & (close > 0.0)
                 & np.isfinite(atr) & (atr > 0.0) & np.isfinite(ma)
                 & np.isfinite(score) & np.isfinite(rho1) & np.isfinite(rho2))
        echo = valid & (rho2 > np.abs(rho1) + params.echo_min)
        long_entry = echo & (score > params.forecast_min) & (close > ma)
        short_entry = echo & (score < -params.forecast_min) & (close < ma)
        raw = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        stop = np.nan
        for i in range(len(data)):
            if position:
                if age == 0:
                    if not (np.isfinite(opening[i]) and opening[i] > 0.0):
                        position = 0
                        continue
                    # Next-open fill, including the YAML's five-basis-point slippage.
                    entry = opening[i] * (1.0 + position * 0.0005)
                    stop = entry - position * params.stop_atr * atr[i - 1]
                age += 1
                stopped = (close[i] < stop if position == 1 else close[i] > stop)
                regime_lost = (close[i] <= ma[i] if position == 1 else close[i] >= ma[i])
                if not valid[i] or stopped or regime_lost or age >= 2:
                    position, age = 0, 0
                raw[i] = position
                continue
            if long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                age = 0
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = EchoParams()
    index = pd.date_range("2020-01-01", periods=8)
    for direction in (1, -1):
        data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
        data.loc[index[1], "open"] = 100.0 + direction * 2.0
        indicators = pd.DataFrame({
            "atr": 1.0, "rho1": 0.0, "rho2": 0.5,
            "echo_score": [direction, 0, 0, -direction, 0, 0, 0, 0],
            "ma200": [100 - direction] * 3 + [100 + direction] * 5,
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        expected = np.array([0, 1, 0, 0, -1, -1, 0, 0]) * direction
        assert np.array_equal(result["signal"].to_numpy(), expected)
        assert (result["size"] > 0.0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
