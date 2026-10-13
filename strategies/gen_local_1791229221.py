from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    vol_window: int = 20
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791229221"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.vol_window + 2, 15)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.vol_window < 5 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("vol_window must be >= 5 and stop_atr must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["variance"] = out["return"].pow(2).rolling(params.vol_window, min_periods=params.vol_window).mean().shift(1)
        out["volatility"] = np.sqrt(out["variance"].where(out["variance"] > 0))
        out["energy"] = out["return"].pow(2).rolling(5, min_periods=5).sum()
        out["budget_used"] = out["energy"] / (5.0 * out["variance"])
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["exposure"] = (0.10 / (np.sqrt(252.0) * out["volatility"])).clip(lower=0.01, upper=0.95)
        out["entry"] = (
            (out["return"] <= -out["volatility"])
            & (out["budget_used"] < 1.0)
            & (out["atr"] > 0)
            & np.isfinite(out["exposure"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        exposure = 1.0
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        exposures = indicators["exposure"].to_numpy(dtype=float)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                active = False
                continue
            if active:
                # This close is observed after the next-open entry fill.
                age += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.stop_atr * atr)
                if close <= stop or age >= 2:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = exposure
                # An exit always emits zero; no same-close re-entry.
                continue
            if entries[i] and np.isfinite(exposures[i]) and exposures[i] > 0:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                exposure = float(exposures[i])
                signals[i] = 1
                sizes[i] = exposure
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"close": [100.0, 101.0, 99.0, 100.0]}, index=index)
    indicators = pd.DataFrame(
        {"atr": [1.0] * 4, "entry": [True, False, True, False], "exposure": [0.4] * 4},
        index=index,
    )
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    assert result["size"].tolist() == [1.0, 0.4, 0.4, 1.0]
    assert strategy.warmup_bars(Params()) >= 22
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
