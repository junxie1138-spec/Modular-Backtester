from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResilientTrendParams:
    return_window: int = 12
    strength_threshold: float = 1.2
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[ResilientTrendParams]):
    strategy_id = "gen_local-w2_1791286372"

    @classmethod
    def params_type(cls):
        return ResilientTrendParams

    def warmup_bars(self, params: ResilientTrendParams) -> int:
        return int(max(200, params.return_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ResilientTrendParams) -> pd.DataFrame:
        if (params.return_window < 2 or params.atr_window < 1
                or params.holding_bars < 1 or params.stop_atr <= 0
                or params.strength_threshold <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        out["return_sum"] = out["return"].rolling(params.return_window).sum()
        out["best_return"] = out["return"].rolling(params.return_window).max().clip(lower=0)
        out["return_energy"] = out["return"].pow(2).rolling(params.return_window).sum()
        out["residual_energy"] = (
            out["return_energy"] - out["best_return"].pow(2)
        ).clip(lower=0)
        out["strength"] = (
            (out["return_sum"] - out["best_return"])
            / np.sqrt(out["residual_energy"].replace(0, np.nan))
        ).replace([np.inf, -np.inf], np.nan)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = data["close"].rolling(200).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ResilientTrendParams,
    ) -> SignalFrame:
        strength = indicators["strength"]
        entry = (
            strength.ge(params.strength_threshold)
            & strength.shift(1).lt(params.strength_threshold)
            & data["close"].gt(indicators["ma200"])
            & indicators["atr"].gt(0)
        ).fillna(False).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        fill_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if active:
                if fill_bar < 0:
                    # The previous close requested entry; this bar supplies its fill open.
                    fill_bar = i
                    stop = opens[i] - params.stop_atr * entry_atr
                if (not np.isfinite(stop) or not np.isfinite(closes[i])
                        or closes[i] < stop
                        or i - fill_bar + 1 >= params.holding_bars):
                    active = False
                # An exit cannot also schedule a new entry on the same close.
            elif entry[i] and np.isfinite(closes[i]) and np.isfinite(atr[i]):
                active = True
                fill_bar = -1
                entry_atr = atr[i]
            raw[i] = int(active)
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=210, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    data.loc[index[201], "close"] = 110.0
    data.loc[index[202], "close"] = 97.0
    data.loc[index[203], "close"] = 95.0
    features = pd.DataFrame(
        {"strength": 0.0, "atr": 2.0, "ma200": 90.0}, index=index
    )
    features.loc[index[199], "strength"] = 2.0
    strategy = GeneratedStrategy()
    params = ResilientTrendParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].iloc[199:205].tolist() == [0, 1, 1, 1, 1, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].iloc[200:208].eq(1).all()
    assert result["signal"].iloc[208] == 0
    assert strategy.warmup_bars(params) >= 200
    computed = strategy.indicators(data, params)
    assert computed.index.equals(data.index)
    assert strategy.generate_signals(data, computed, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
