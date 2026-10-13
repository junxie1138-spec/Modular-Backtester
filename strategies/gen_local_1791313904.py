from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    correlation_window: int = 42
    echo_threshold: float = 0.25
    forecast_threshold: float = 0.35
    profit_target: float = 0.012
    max_hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local_1791313904"

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    @staticmethod
    def warmup_bars(params: EchoParams) -> int:
        if not (
            params.correlation_window >= 6
            and 0 < params.echo_threshold < 1
            and params.forecast_threshold > 0
            and params.profit_target > 0
            and 3 <= params.max_hold_bars <= 5
            and 0 < params.min_size <= params.max_size <= 1
        ):
            raise ValueError("Invalid strategy parameters")
        return int(params.correlation_window + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None)
        acceleration = returns.diff()
        lag1 = acceleration.shift(1)
        lag2 = acceleration.shift(2)
        lag3 = acceleration.shift(3)
        echo = acceleration.rolling(params.correlation_window).corr(lag3)
        adjacent = acceleration.rolling(params.correlation_window).corr(lag1)
        noise = returns.rolling(params.correlation_window).std(ddof=0)
        predicted_acceleration = echo * lag2
        predicted_return = returns + predicted_acceleration
        score = predicted_return / noise.where(noise > 0)
        eligible = (
            (echo > params.echo_threshold)
            & (adjacent < 0)
            & (returns <= 0)
            & (acceleration <= 0)
            & (lag2 > 0)
            & (score > params.forecast_threshold)
        )
        conviction = score.clip(lower=0) / (1 + score.clip(lower=0))
        return pd.DataFrame({
            "return": returns,
            "acceleration": acceleration,
            "acceleration_lag1": lag1,
            "acceleration_lag2": lag2,
            "acceleration_lag3": lag3,
            "echo_correlation": echo,
            "adjacent_correlation": adjacent,
            "return_noise": noise,
            "predicted_acceleration": predicted_acceleration,
            "predicted_return": predicted_return,
            "score": score,
            "conviction": conviction,
            "eligible": eligible,
        }, index=data.index).replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EchoParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].fillna(0).to_numpy(dtype=float)
        entry_bar = -1
        entry_price = 0.0
        held_size = params.min_size
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    entry_bar = -1
                    continue
                # Targets are observed at the close; exits fill next open.
                target_hit = closes[i] >= entry_price * (1 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_up:
                    entry_bar = -1
                    continue
                signal[i] = 1
                sizes[i] = held_size
            elif eligible[i] and np.isfinite(closes[i]) and closes[i] > 0:
                strength = float(np.clip(conviction[i], 0, 1))
                held_size = params.min_size + (params.max_size - params.min_size) * strength
                entry_bar = i + 1
                signal[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = EchoParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    data.loc[index[2], "close"] = 102.0
    test_indicators = pd.DataFrame({"eligible": False, "conviction": 0.0}, index=index)
    test_indicators.loc[index[[0, 6]], "eligible"] = True
    test_indicators.loc[index[0], "conviction"] = 0.5
    test_indicators.loc[index[6], "conviction"] = 0.75
    result = GeneratedStrategy.generate_signals(data, test_indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert np.isclose(result["size"].iloc[1], 0.60)
    assert np.isclose(result["size"].iloc[7], 0.775)
    flat = data.copy()
    flat["close"] = 100.0
    flat = pd.concat([flat] * 6, ignore_index=True)
    computed = GeneratedStrategy.indicators(flat, params)
    assert np.isfinite(computed.to_numpy(dtype=float)).all()
    assert not computed["eligible"].any()


if __name__ == "__main__":
    _self_check()
