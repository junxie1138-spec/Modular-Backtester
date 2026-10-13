from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnTransitionParams:
    regime_window: int = 20
    impulse_window: int = 3
    persistence_on: float = 0.58
    reversal_on: float = 0.42
    entry_z: float = 1.0
    profit_target: float = 0.018
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ReturnTransitionParams]):
    strategy_id = "gen_local_1791268897"

    @classmethod
    def params_type(cls) -> type[ReturnTransitionParams]:
        return ReturnTransitionParams

    def warmup_bars(self, params: ReturnTransitionParams) -> int:
        return int(max(params.regime_window + 2, params.impulse_window + 1))

    def indicators(self, data: pd.DataFrame, params: ReturnTransitionParams) -> pd.DataFrame:
        if params.regime_window < 2 or params.impulse_window < 1:
            raise ValueError("Lookback windows must be positive; regime_window must be at least two.")
        if not 0.0 < params.reversal_on < params.persistence_on < 1.0:
            raise ValueError("Regime thresholds must satisfy 0 < reversal_on < persistence_on < 1.")
        if params.entry_z <= 0.0 or params.profit_target <= 0.0 or params.hold_bars < 1:
            raise ValueError("Entry threshold, profit target, and holding period must be positive.")
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        signs = np.sign(returns)
        previous_signs = signs.shift(1)
        valid_pair = signs.notna() & previous_signs.notna() & signs.ne(0) & previous_signs.ne(0)
        repeat = signs.eq(previous_signs).astype(float).where(valid_pair)
        persistence = repeat.rolling(params.regime_window, min_periods=params.regime_window).mean()
        volatility = returns.rolling(params.regime_window, min_periods=params.regime_window).std(ddof=0).replace(0.0, np.nan)
        impulse = returns.rolling(params.impulse_window, min_periods=params.impulse_window).sum()
        result = pd.DataFrame(index=data.index)
        result["return"] = returns
        result["persistence"] = persistence
        result["volatility"] = volatility
        result["impulse"] = impulse
        result["continuation_score"] = impulse / (volatility * np.sqrt(params.impulse_window))
        result["reversal_score"] = -returns / volatility
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReturnTransitionParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        persistence = indicators["persistence"].to_numpy(dtype=float)
        continuation = indicators["continuation_score"].to_numpy(dtype=float)
        reversal = indicators["reversal_score"].to_numpy(dtype=float)
        regime = 0
        position = 0
        pending_direction = 0
        entry_price = 0.0
        entry_bar = -1
        position_size = 1.0
        pending_size = 1.0

        for i in range(count):
            if pending_direction:
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    position = pending_direction
                    entry_price = opens[i]
                    entry_bar = i
                    position_size = pending_size
                pending_direction = 0

            if np.isfinite(persistence[i]):
                if persistence[i] >= params.persistence_on:
                    regime = 1
                elif persistence[i] <= params.reversal_on:
                    regime = -1

            if position:
                gain = position * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                time_hit = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_hit:
                    position = 0
                else:
                    signals[i] = position
                    sizes[i] = position_size
                continue

            if regime == 0 or not np.isfinite(persistence[i]):
                continue
            score = continuation[i] if regime == 1 else reversal[i]
            if not np.isfinite(score) or abs(score) < params.entry_z:
                continue
            conviction = float(np.clip(abs(score) / (2.0 * params.entry_z), 0.0, 1.0))
            pending_direction = 1 if score > 0.0 else -1
            pending_size = params.min_size + (params.max_size - params.min_size) * conviction
            signals[i] = pending_direction
            sizes[i] = pending_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = ReturnTransitionParams(hold_bars=4)
    index = pd.date_range("2020-01-01", periods=7)
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
        features = pd.DataFrame({"persistence": 0.7, "continuation_score": [1.5 * direction, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "reversal_score": 0.0}, index=index)
        result = strategy.generate_signals(data, features, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0, 0]
        assert params.min_size < result["size"].iloc[1] < params.max_size
        data.loc[index[2], "close"] = 103.0 if direction == 1 else 97.0
        result = strategy.generate_signals(data, features, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0, 0, 0]
    assert strategy.warmup_bars(params) >= params.regime_window + 2


if __name__ == "__main__":
    _self_check()
