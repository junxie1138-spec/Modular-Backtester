from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    anchor_bars: int = 20
    volatility_bars: int = 8
    min_displacement: float = 0.35
    max_displacement: float = 1.75
    profit_target: float = 0.012
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791285794"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.anchor_bars, params.volatility_bars) + 3

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["stress_weight"] = out["true_range"].pow(2)
        out["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3.0
        weights = out["stress_weight"].rolling(
            params.anchor_bars, min_periods=params.anchor_bars
        ).sum()
        out["anchor"] = (out["typical_price"] * out["stress_weight"]).rolling(
            params.anchor_bars, min_periods=params.anchor_bars
        ).sum().div(weights.replace(0.0, np.nan))
        out["atr"] = out["true_range"].rolling(
            params.volatility_bars, min_periods=params.volatility_bars
        ).mean()
        out["return"] = data["close"].pct_change(fill_method=None)
        out["downside_variance"] = out["return"].clip(upper=0.0).pow(2)
        out["downside_volatility"] = np.sqrt(out["downside_variance"].rolling(
            params.volatility_bars, min_periods=params.volatility_bars
        ).mean())
        out["displacement"] = (out["anchor"] - data["close"]).div(
            out["atr"].replace(0.0, np.nan)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        out["recovery"] = (
            out["displacement"].between(params.min_displacement, params.max_displacement)
            & (out["downside_volatility"] < out["downside_volatility"].shift(1))
            & (data["close"] > previous_close)
        ).fillna(False)
        out["confirmed"] = out["recovery"] & out["recovery"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        pending = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            # A decision from the previous close fills at this bar's open.
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    continue
                entry_bar = i
                entry_price = opens[i]
            if entry_bar >= 0:
                # Close-based exits become flat at the following bar's open.
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    decisions[i] = 1
            elif confirmed[i]:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"confirmed": [False, True, False, False, False, False, False, False]},
        index=index,
    )
    params = Params()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    data.loc[index[3], "close"] = 102.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    assert result["size"].gt(0.0).all()
    derived = GeneratedStrategy.indicators(None, data, params)
    assert not derived["confirmed"].any()


if __name__ == "__main__":
    _self_check()
