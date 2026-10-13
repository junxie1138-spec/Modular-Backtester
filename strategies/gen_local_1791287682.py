from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityWedgeParams:
    atr_window: int = 14
    volatility_window: int = 20
    percentile_window: int = 126
    entry_percentile: float = 0.80
    target_volatility: float = 0.12
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[VolatilityWedgeParams]):
    strategy_id = "gen_local_1791287682"

    @classmethod
    def params_type(cls) -> type[VolatilityWedgeParams]:
        return VolatilityWedgeParams

    def warmup_bars(self, params: VolatilityWedgeParams) -> int:
        return int(max(params.atr_window, params.volatility_window + 1) + params.percentile_window)

    def indicators(self, data: pd.DataFrame, params: VolatilityWedgeParams) -> pd.DataFrame:
        if min(params.atr_window, params.percentile_window, params.holding_bars) < 1 or params.volatility_window < 2:
            raise ValueError("Windows and holding period must be positive; volatility window must be at least two.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("Entry percentile must be between zero and one.")
        if not np.isfinite(params.target_volatility) or not np.isfinite(params.stop_atr) or min(params.target_volatility, params.stop_atr) <= 0:
            raise ValueError("Target volatility and stop multiplier must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["daily_volatility"] = out["return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        out["annual_volatility"] = out["daily_volatility"] * np.sqrt(252.0)
        out["atr_fraction"] = out["atr"] / data["close"].where(data["close"] > 0)
        out["wedge"] = (out["atr_fraction"] / out["daily_volatility"].where(out["daily_volatility"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["wedge_threshold"] = out["wedge"].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_percentile)
        out["momentum"] = data["close"].pct_change(params.volatility_window, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["wedge"] > out["wedge_threshold"])
            & (out["momentum"] > 0)
            & (out["atr"] > 0)
            & (out["annual_volatility"] > 0)
            & (data["close"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilityWedgeParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        pending = False
        entry_bar = -1
        frozen_atr = np.nan
        stop_price = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_bar = i
                    stop_price = opens[i] - params.stop_atr * frozen_atr
            if active:
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                expired = i - entry_bar + 1 >= params.holding_bars
                if stopped or expired:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                frozen_atr = atr[i]
                pending = True
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = (params.target_volatility / indicators["annual_volatility"].where(indicators["annual_volatility"] > 0)).replace([np.inf, -np.inf], np.nan).clip(lower=0.05, upper=0.95).shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"open": [100.0, 110.0, 101.0, 101.0], "high": [101.0, 111.0, 102.0, 102.0], "low": [99.0, 100.0, 100.0, 100.0], "close": [100.0, 101.0, 101.0, 101.0], "volume": [1000.0] * 4}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False], "atr": [1.0] * 4, "annual_volatility": [0.20] * 4}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, VolatilityWedgeParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0]
    assert result["size"].gt(0).all()
