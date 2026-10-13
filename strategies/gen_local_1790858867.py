from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    window: int = 5
    efficiency: float = 0.65
    max_share: float = 0.50
    min_net_return: float = 0.004
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = "gen_local_1790858867"

    @classmethod
    def params_type(cls):
        return MomentumParams

    def warmup_bars(self, params: MomentumParams) -> int:
        if params.window < 2 or params.max_hold_bars not in (1, 2):
            raise ValueError("window must be >= 2 and max_hold_bars must be 1 or 2")
        values = [params.efficiency, params.max_share,
                  params.min_net_return, params.profit_target]
        if not np.isfinite(values).all():
            raise ValueError("Thresholds must be finite")
        if not (0 < params.efficiency <= 1 and 0 < params.max_share <= 1
                and params.min_net_return > 0 and params.profit_target > 0):
            raise ValueError("Invalid momentum or exit thresholds")
        return int(max(200, params.window + 1))

    def indicators(self, data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["abs_return"] = out["return"].abs()
        out["net_return"] = out["return"].rolling(params.window).sum()
        out["gross_return"] = out["abs_return"].rolling(params.window).sum()
        out["largest_return"] = out["abs_return"].rolling(params.window).max()
        denominator = out["gross_return"].where(out["gross_return"] > 0)
        out["efficiency"] = out["net_return"].abs() / denominator
        out["concentration"] = out["largest_return"] / denominator
        out["ma200"] = close.rolling(200).mean()
        eligible = (
            (out["efficiency"] >= params.efficiency)
            & (out["concentration"] <= params.max_share)
            & (out["net_return"].abs() >= params.min_net_return)
        )
        long_setup = (eligible & (out["net_return"] > 0)
                      & (out["return"] > 0) & (close > out["ma200"]))
        short_setup = (eligible & (out["net_return"] < 0)
                       & (out["return"] < 0) & (close < out["ma200"]))
        out["setup"] = np.select([long_setup, short_setup], [1, -1], default=0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MomentumParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        signals = np.zeros(len(data), dtype=int)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # Read the fill proxy only when its opening bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                valid = (np.isfinite(entry_price) and entry_price > 0
                         and np.isfinite(closes[i]) and closes[i] > 0)
                gain = position * (closes[i] / entry_price - 1) if valid else 0.0
                age = i - entry_bar + 1
                if not valid or gain >= params.profit_target or age >= params.max_hold_bars:
                    position = 0
                    continue
                signals[i] = position
            elif setups[i] in (-1, 1):
                position = int(setups[i])
                entry_bar = i + 1
                entry_price = np.nan
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = MomentumParams()
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 98.0,
        "close": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    setup = pd.DataFrame({"setup": [1, 0, -1, 0, 0, 0]}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, setup, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(
            data.iloc[:end], setup.iloc[:end], ctx, params).data
        assert prefix.equals(result.iloc[:end])
    flat = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0,
        "close": 100.0, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=210, freq="D"))
    assert strategy.indicators(flat, params)["setup"].eq(0).all()


if __name__ == "__main__":
    _self_check()
