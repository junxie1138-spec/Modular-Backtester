from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnCancellationParams:
    balance_window: int = 5
    volatility_window: int = 20
    max_efficiency: float = 0.25
    shock_threshold: float = 1.5
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ReturnCancellationParams]):
    strategy_id = "gen_local-w2_1791316737"

    @classmethod
    def params_type(cls) -> type[ReturnCancellationParams]:
        return ReturnCancellationParams

    def warmup_bars(self, params: ReturnCancellationParams) -> int:
        if (
            min(params.balance_window, params.volatility_window) < 2
            or not 0.0 <= params.max_efficiency <= 1.0
            or not np.isfinite(params.shock_threshold)
            or params.shock_threshold <= 0.0
            or not np.isfinite(params.profit_target)
            or params.profit_target <= 0.0
            or params.max_hold_bars not in (1, 2)
        ):
            raise ValueError("Invalid return-cancellation parameters")
        return max(200, params.balance_window + 2, params.volatility_window + 2)

    def indicators(self, data: pd.DataFrame, params: ReturnCancellationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return"] = returns
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["prior_net"] = returns.rolling(params.balance_window, min_periods=params.balance_window).sum().shift(1)
        out["prior_gross"] = returns.abs().rolling(params.balance_window, min_periods=params.balance_window).sum().shift(1)
        out["efficiency"] = out["prior_net"].abs() / out["prior_gross"].replace(0.0, np.nan)
        out["prior_volatility"] = returns.rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0).shift(1)
        out["shock"] = returns / out["prior_volatility"].replace(0.0, np.nan)
        ready = (out["efficiency"] <= params.max_efficiency) & np.isfinite(out["shock"])
        out["entry"] = np.select(
            [
                ready & (out["shock"] <= -params.shock_threshold) & (data["close"] > out["ma200"]),
                ready & (out["shock"] >= params.shock_threshold) & (data["close"] < out["ma200"]),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReturnCancellationParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0.0 and np.isfinite(closes[i])
                gain = position * (closes[i] / entry_price - 1.0) if valid else 0.0
                # Targets are observed at the close; the shifted exit fills next open.
                if not valid or gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    position = 0
                    continue
            elif entries[i] != 0 and i + 1 < len(data):
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            desired[i] = position
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = ReturnCancellationParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0,
         "close": [100.0, 101.0, 100.0, 100.0, 99.0, 100.0, 100.0], "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0.0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
