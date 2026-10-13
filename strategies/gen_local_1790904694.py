from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volatility_window: int = 8
    atr_window: int = 20
    continuation_ratio: float = 0.70
    reversal_ratio: float = 0.45
    shock_threshold: float = 0.80
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790904694"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.volatility_window, params.atr_window) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.volatility_window >= 2 and params.atr_window >= 2
                and 0 < params.reversal_ratio < params.continuation_ratio
                and params.shock_threshold > 0 and params.profit_target > 0
                and 1 <= params.max_hold_bars <= 2):
            raise ValueError("Invalid volatility, regime, target, or holding parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["previous_close"] = close.shift(1)
        out["return"] = close.div(out["previous_close"]).sub(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["volatility"] = out["return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["prior_volatility"] = out["volatility"].shift(1).replace(0, np.nan)
        out["prior_atr_fraction"] = out["atr"].shift(1).div(
            out["previous_close"]
        ).replace(0, np.nan)
        out["regime_ratio"] = out["prior_volatility"].div(out["prior_atr_fraction"])
        out["shock"] = out["return"].div(out["prior_volatility"])
        out = out.replace([np.inf, -np.inf], np.nan)
        ratio = out["regime_ratio"]
        out["regime"] = np.select(
            [ratio >= params.continuation_ratio, ratio <= params.reversal_ratio],
            [1, -1], default=0
        ).astype(int)
        out["regime_strength"] = pd.Series(np.where(
            out["regime"] == 1,
            ratio.div(params.continuation_ratio),
            params.reversal_ratio / ratio.replace(0, np.nan)
        ), index=data.index).replace([np.inf, -np.inf], np.nan)
        out["conviction"] = (
            out["shock"].abs().div(params.shock_threshold)
            * out["regime_strength"]
        ).replace([np.inf, -np.inf], np.nan)
        eligible = (
            out["shock"].abs().ge(params.shock_threshold)
            & out["regime"].ne(0)
            & out["conviction"].notna()
        )
        out["candidate"] = (
            np.sign(out["shock"].fillna(0)) * out["regime"]
        ).where(eligible, 0).astype(int)
        out["entry_size"] = (
            0.25 * out["conviction"]
        ).clip(lower=0.25, upper=0.95).fillna(0.25)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        candidate = indicators["candidate"].to_numpy(dtype=int)
        entry_size = indicators["entry_size"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.25, dtype=float)
        side, decision_bar = 0, -1
        entry_price, position_size = np.nan, 0.25
        for i in range(len(data)):
            if side:
                if i == decision_bar + 1:
                    entry_price = opens[i]
                valid = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i]) and closes[i] > 0
                )
                # Targets are checked at the close; exits fill next bar.
                gain = side * (closes[i] / entry_price - 1) if valid else 0.0
                if (not valid or gain >= params.profit_target
                        or i - decision_bar >= params.max_hold_bars):
                    side = 0
                    entry_price = np.nan
                    continue
            elif (candidate[i] and np.isfinite(closes[i]) and closes[i] > 0
                    and np.isfinite(entry_size[i]) and entry_size[i] > 0):
                side = int(candidate[i])
                decision_bar = i
                position_size = float(entry_size[i])
            signal[i] = side
            if side:
                size[i] = position_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 98.0,
        "close": [100.0, 100.0, 100.0, 100.0, 99.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "candidate": [1, 0, 0, -1, 0, 0],
        "entry_size": [0.4, 0.4, 0.4, 0.6, 0.6, 0.6],
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, Params()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    assert np.allclose(result["size"], [0.25, 0.4, 0.4, 0.25, 0.6, 0.25])
    data.loc[index[1], "close"] = 101.0
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, Params()
    ).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0]


if __name__ == "__main__":
    _self_check()
