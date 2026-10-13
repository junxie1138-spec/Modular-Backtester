from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 5
    prediction_window: int = 3
    cohort_window: int = 10
    percentile_window: int = 63
    entry_percentile: float = 0.85
    volatility_window: int = 20
    target_volatility: float = 0.12
    profit_target: float = 0.0075
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791294371"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.slope_window + params.prediction_window,
            params.cohort_window + 2,
            params.volatility_window + 1,
        ) + params.percentile_window)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.slope_window < 2 or params.prediction_window < 1
                or params.cohort_window < 1 or params.percentile_window < 2
                or params.volatility_window < 2
                or not 0.0 < params.entry_percentile < 1.0
                or params.target_volatility <= 0.0
                or params.profit_target <= 0.0
                or params.max_hold_bars not in (1, 2)):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["clock"] = np.arange(len(data), dtype=float)
        # Rolling OLS slope is the causal latent trend estimate.
        out["slope"] = out["log_close"].rolling(params.slope_window).cov(
            out["clock"], ddof=0
        ) / ((params.slope_window ** 2 - 1) / 12.0)
        out["predicted_slope"] = out["slope"].shift(1).rolling(
            params.prediction_window
        ).mean()
        out["volatility"] = out["return"].rolling(
            params.volatility_window
        ).std(ddof=0)
        out["innovation"] = (
            out["slope"] - out["predicted_slope"]
        ) / out["volatility"].replace(0.0, np.nan)
        out["down_day"] = (out["return"] < 0).astype(float).where(
            out["return"].notna()
        )
        out["susceptible_cohort"] = out["down_day"].shift(1).rolling(
            params.cohort_window
        ).mean()
        out["ignition"] = out["innovation"] * out["susceptible_cohort"]
        out["threshold"] = out["ignition"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.entry_percentile)
        out["exposure"] = (
            params.target_volatility / (out["volatility"] * np.sqrt(252.0))
        ).replace([np.inf, -np.inf], np.nan).clip(0.05, 0.95).fillna(0.05)
        out["entry"] = (
            (out["ignition"] > out["threshold"])
            & (out["ignition"] > 0.0)
            & (out["predicted_slope"] <= 0.0)
            & (out["slope"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.95, dtype=float)
        eligible = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        exposure = indicators["exposure"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        frozen_size = 0.95
        for i in range(n):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                size[i] = frozen_size
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    entry_bar = -1
                elif (
                    (np.isfinite(closes[i])
                     and closes[i] >= entry_price * (1.0 + params.profit_target))
                    or i - entry_bar + 1 >= params.max_hold_bars
                ):
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif eligible[i] and np.isfinite(exposure[i]) and exposure[i] > 0.0:
                # Today's close decision enters at the next bar's open.
                entry_bar = i + 1
                frozen_size = float(np.clip(exposure[i], 0.05, 0.95))
                signal[i] = 1
                size[i] = frozen_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame(
        {"entry": [True, False, False, False, False], "exposure": 0.4},
        index=index,
    )
    strategy = GeneratedStrategy()
    params = Params()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert np.allclose(result["size"].iloc[1:3], 0.4)
    data.loc[index[1], "close"] = 101.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
