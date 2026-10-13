from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPropagationParams:
    propagation_threshold: float = 0.25
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[GapPropagationParams]):
    strategy_id = "gen_local-w2_1791308608"

    @classmethod
    def params_type(cls):
        return GapPropagationParams

    def warmup_bars(self, params: GapPropagationParams) -> int:
        return 42

    def indicators(self, data: pd.DataFrame, params: GapPropagationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        prior = close.shift(1)
        out["gap"] = data["open"] / prior.where(prior > 0) - 1.0
        out["lagged_gap"] = out["gap"].shift(1)
        out["intraday"] = close / data["open"].where(data["open"] > 0) - 1.0
        # Rolling correlation: [pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.Rolling.corr.html)
        out["propagation"] = out["lagged_gap"].rolling(40, min_periods=40).corr(out["intraday"])
        out["change"] = close.diff()
        out["displacement"] = close.diff(20)
        out["path_length"] = out["change"].abs().rolling(20, min_periods=20).sum()
        out["efficiency"] = out["displacement"] / out["path_length"].replace(0.0, np.nan)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - prior).abs(), (data["low"] - prior).abs()],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapPropagationParams,
    ) -> SignalFrame:
        if not np.isfinite(params.propagation_threshold) or not 0.0 <= params.propagation_threshold <= 1.0:
            raise ValueError("propagation_threshold must be between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be positive")
        entry = (
            (indicators["gap"] > 0.0)
            & (indicators["propagation"] > params.propagation_threshold)
            & (indicators["efficiency"] > 0.35)
            & (indicators["atr"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    active = True
                    # Entry uses the next open and the config's fixed five-basis-point slippage.
                    stop = opens[i] * 1.0005 - params.stop_atr * atr[i - 1]
                    age = 0
            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= 8:
                    active = False
                else:
                    decisions[i] = 1
            elif entry[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapPropagationParams()
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 121.0, "low": 94.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"gap": 0.0, "propagation": 0.8, "efficiency": 0.8, "atr": 2.0}, index=index
    )
    indicators.iloc[0, indicators.columns.get_loc("gap")] = 0.01
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=42)
    data.iloc[2, data.columns.get_loc("close")] = 120.0
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    data.iloc[3, data.columns.get_loc("close")] = 96.0
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1] + [0] * 8
    assert (signals["size"] > 0.0).all()
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    derived = strategy.indicators(flat, params)
    assert not strategy.generate_signals(flat, derived, ctx, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
