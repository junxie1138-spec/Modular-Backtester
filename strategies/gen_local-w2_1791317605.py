from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DeformationParams:
    atr_period: int = 14
    percentile_window: int = 126
    stress_quantile: float = 0.85
    relief_quantile: float = 0.50
    retention: float = 0.60
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[DeformationParams]):
    strategy_id = "gen_local-w2_1791317605"

    @classmethod
    def params_type(cls) -> type[DeformationParams]:
        return DeformationParams

    @staticmethod
    def warmup_bars(params: DeformationParams) -> int:
        return int(params.atr_period + params.percentile_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: DeformationParams) -> pd.DataFrame:
        if not (
            params.atr_period >= 2 and params.percentile_window >= 2
            and 0 < params.relief_quantile < params.stress_quantile < 1
            and 0 < params.retention <= 1
            and params.breakeven_pct > 0 and params.trail_atr > 0
            and params.max_hold >= 1
        ):
            raise ValueError("Invalid strategy parameters")
        result = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        result["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        result["atr"] = result["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        result["normalized_atr"] = result["atr"] / data["close"].where(data["close"] > 0)
        history = result["normalized_atr"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        )
        result["stress_threshold"] = history.quantile(params.stress_quantile)
        result["relief_threshold"] = history.quantile(params.relief_quantile)
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: DeformationParams
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        active = pending = stressed = armed = False
        anchor = peak = entry = stop = 0.0
        entry_bar = 0
        values = indicators[
            ["atr", "normalized_atr", "stress_threshold", "relief_threshold"]
        ].to_numpy(dtype=float)
        prices = data[["open", "low", "close"]].to_numpy(dtype=float)
        for i in range(len(data)):
            opening, low, close = prices[i]
            atr, vol, upper, lower = values[i]
            if not np.isfinite([opening, low, close, atr, vol, upper, lower]).all() or min(opening, low, close, atr) <= 0:
                active = pending = stressed = False
                continue
            if pending:
                # The previous close's decision fills at this bar's open.
                entry = opening
                stop = entry - params.trail_atr * values[i - 1, 0]
                entry_bar = i
                active, pending, armed = True, False, False
            if active:
                # Test the previously established stop before raising it.
                # ponytail: stop breaches exit next open; use per-bar execution for intraday stop fills.
                if low <= stop or i - entry_bar + 1 >= params.max_hold:
                    active = stressed = False
                    continue
                if close >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, close - params.trail_atr * atr)
                if close <= stop:
                    active = stressed = False
                    continue
                raw[i] = 1
                continue
            if not stressed and i > 0 and vol > upper:
                anchor = prices[i - 1, 2]
                peak = close
                stressed = bool(np.isfinite(anchor) and anchor > 0)
            if stressed:
                peak = max(peak, close)
                if vol <= lower:
                    stressed = False
                    if peak > anchor and close >= anchor + params.retention * (peak - anchor):
                        pending = True
                        raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100, 100, 104, 103, 105, 108, 107, 104],
        "high": [101, 105, 104, 106, 109, 109, 108, 105],
        "low": [99, 99, 102, 102, 104, 106, 104, 103],
        "close": [100, 104, 103, 105, 108, 107, 104, 104],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 1, 1, 1, 10, 1, 1],
        "normalized_atr": [1, 3, 1, 1, 1, 1, 1, 1],
        "stress_threshold": 2.0,
        "relief_threshold": 1.5,
    }, index=index)
    params = DeformationParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    params.max_hold = 2
    assert GeneratedStrategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    params.max_hold = 20
    for n in range(1, len(data) + 1):
        prefix = GeneratedStrategy.generate_signals(data.iloc[:n], indicators.iloc[:n], None, params).data
        assert prefix.equals(result.iloc[:n])


if __name__ == "__main__":
    _self_check()
