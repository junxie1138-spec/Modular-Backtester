from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownEchoParams:
    peak_window: int = 63
    correlation_window: int = 40
    min_depth: float = 0.03
    min_anticorrelation: float = 0.15
    min_echo: float = 0.10
    holding_bars: int = 8
    min_size: float = 0.15
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DrawdownEchoParams]):
    strategy_id = "gen_local-w2_1791208975"

    @classmethod
    def params_type(cls) -> type[DrawdownEchoParams]:
        return DrawdownEchoParams

    def warmup_bars(self, params: DrawdownEchoParams) -> int:
        return int(params.peak_window + params.correlation_window + 3)

    def indicators(self, data: pd.DataFrame, params: DrawdownEchoParams) -> pd.DataFrame:
        if not (
            params.peak_window >= 2 and params.correlation_window >= 3
            and params.holding_bars >= 1 and 0 < params.min_depth < 1
            and 0 < params.min_anticorrelation < 1 and 0 < params.min_echo < 1
            and 0 < params.min_size <= params.max_size <= 1
        ):
            raise ValueError("Invalid drawdown echo parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = 1.0 - close / out["peak"]
        out["depth_change"] = out["depth"].diff()
        out["lag_one"] = out["depth_change"].shift(1)
        out["lag_two"] = out["depth_change"].shift(2)
        window = out["depth_change"].rolling(
            params.correlation_window, min_periods=params.correlation_window
        )
        out["rho_one"] = window.corr(out["lag_one"]).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
        out["rho_two"] = window.corr(out["lag_two"]).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
        out["strength"] = (
            (-out["rho_one"]).clip(lower=0)
            * out["rho_two"].clip(lower=0)
            * out["depth"] / params.min_depth
        ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
        out["entry_size"] = (
            params.min_size + (params.max_size - params.min_size)
            * (out["strength"] / 0.15).clip(0, 1)
        )
        out["eligible"] = (
            (out["depth"] >= params.min_depth)
            & (out["depth_change"] > 0)
            & (out["rho_one"] <= -params.min_anticorrelation)
            & (out["rho_two"] >= params.min_echo)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: DrawdownEchoParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        candidates = indicators["entry_size"].fillna(params.min_size).to_numpy(dtype=float)
        exit_bar = -1
        entry_size = params.min_size
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                    sizes[i] = entry_size
                else:
                    exit_bar = -1
            elif eligible[i]:
                exit_bar = i + params.holding_bars
                entry_size = float(np.clip(candidates[i], params.min_size, params.max_size))
                signals[i] = 1
                sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = DrawdownEchoParams()
    index = pd.date_range("2020-01-01", periods=22, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"eligible": True, "entry_size": np.linspace(0.2, 0.9, len(index))},
        index=index,
    )
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[:10].tolist() == [0] + [1] * 8 + [0]
    assert result["signal"].iloc[10:19].tolist() == [1] * 8 + [0]
    assert (result["size"].iloc[1:9] == 0.2).all()
    assert result["size"].gt(0).all()
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
        index=pd.date_range("2020-01-01", periods=200, freq="B"),
    )
    assert not strategy.indicators(flat, params)["eligible"].any()


if __name__ == "__main__":
    _self_check()
