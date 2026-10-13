from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    lookback: int = 20
    profit_target: float = 0.008


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local-w2_1791332981"

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    @staticmethod
    def warmup_bars(params: EchoParams) -> int:
        return int(2 * max(5, params.lookback) + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if params.lookback < 5 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("lookback must be >= 5 and profit_target must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["volatility"] = out["return"].rolling(params.lookback).std()
        out["short_volatility"] = out["return"].rolling(5).std()
        out["lagged_volatility"] = out["volatility"].shift(1)
        out["standardized_return"] = out["return"] / out["lagged_volatility"].replace(0.0, np.nan)
        out["lag_one_return"] = out["standardized_return"].shift(1)
        out["lag_two_return"] = out["standardized_return"].shift(2)
        out["rho_one"] = out["standardized_return"].rolling(params.lookback).corr(out["lag_one_return"])
        out["rho_two"] = out["standardized_return"].rolling(params.lookback).corr(out["lag_two_return"])
        out = out.replace([np.inf, -np.inf], np.nan)
        out["setup"] = (
            (out["short_volatility"] < 0.85 * out["volatility"])
            & (out["rho_two"] > 0.20)
            & (out["rho_one"] < 0.05)
            & (out["rho_two"] - out["rho_one"] > 0.25)
            & (out["lag_one_return"] > 0.75)
            & (out["standardized_return"] < 0.0)
            & (out["standardized_return"] > -0.75)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        setup = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar < 0:
                if setup[i] and np.isfinite(closes[i]) and closes[i] > 0:
                    decisions[i] = 1
                    entry_bar = i + 1
            else:
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                # Targets are observed at the close; exits execute next bar.
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or age >= 2:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0] * 7,
        "high": [103.0] * 7,
        "low": [99.0] * 7,
        "close": [100.0, 102.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=index)
    setups = pd.DataFrame({"setup": [True, False, True, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(data, setups, None, EchoParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(EchoParams()) == 43
    assert not GeneratedStrategy.indicators(data, EchoParams())["setup"].any()


if __name__ == "__main__":
    _self_check()
