from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ordinal_window: int = 36
    correlation_window: int = 36
    entropy_ceiling: float = 0.92
    min_forecast: float = 0.10
    atr_window: int = 14
    stop_atr: float = 1.25
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790937691"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if min(params.ordinal_window, params.correlation_window, params.atr_window) < 3:
            raise ValueError("Lookbacks must be at least three bars")
        if not 0.0 <= params.entropy_ceiling <= 1.0:
            raise ValueError("Entropy ceiling must be between zero and one")
        if params.min_forecast < 0 or params.stop_atr <= 0 or params.entry_slippage_bps < 0:
            raise ValueError("Invalid forecast, stop, or slippage parameter")
        return int(max(params.ordinal_window + 3, params.correlation_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = returns.shift(2), returns.shift(1), returns
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns are ordered chronologically for deterministic patterns.
        conditions = [
            (a <= b) & (b <= c),
            (a <= c) & (c < b),
            (b < a) & (a <= c),
            (b <= c) & (c < a),
            (c < a) & (a <= b),
            (c < b) & (b < a),
        ]
        pattern = pd.Series(np.select(conditions, range(6), default=0), index=data.index).where(valid)
        out["return"] = returns
        out["ordinal_pattern"] = pattern
        entropy = pd.Series(0.0, index=data.index)
        for code in range(6):
            probability = pattern.eq(code).astype(float).where(valid).rolling(
                params.ordinal_window, min_periods=params.ordinal_window
            ).mean()
            out[f"pattern_probability_{code}"] = probability
            entropy -= probability * np.log(probability.where(probability > 0, 1.0))
        out["ordinal_entropy"] = entropy / np.log(6.0)
        lag = returns.shift(1)
        current_window = returns.rolling(params.correlation_window, min_periods=params.correlation_window)
        lag_window = lag.rolling(params.correlation_window, min_periods=params.correlation_window)
        out["return_mean"] = current_window.mean()
        out["lag_mean"] = lag_window.mean()
        out["return_std"] = current_window.std().replace(0.0, np.nan)
        out["lag_variance"] = lag_window.var().replace(0.0, np.nan)
        out["lag_covariance"] = current_window.cov(lag)
        out["autocorrelation"] = current_window.corr(lag).clip(-1.0, 1.0)
        out["beta"] = out["lag_covariance"] / out["lag_variance"]
        out["forecast"] = out["return_mean"] + out["beta"] * (returns - out["lag_mean"])
        out["forecast_score"] = out["forecast"] / out["return_std"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(
            previous_close.notna() & data["high"].notna() & data["low"].notna()
        )
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["complexity_agrees"] = out["ordinal_entropy"].le(params.entropy_ceiling)
        out["rebound_agrees"] = (
            out["autocorrelation"].lt(-0.10)
            & returns.lt(0.0)
            & out["forecast_score"].gt(params.min_forecast)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        ready = indicators[["ordinal_entropy", "autocorrelation", "forecast_score", "atr"]].notna().all(axis=1)
        setup = (
            ready
            & indicators["complexity_agrees"].fillna(False)
            & indicators["rebound_agrees"].fillna(False)
            & indicators["atr"].gt(0.0)
        ).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        holding = False
        age = 0
        fixed_stop = np.nan
        for i in range(len(data)):
            # Yesterday's decision becomes today's position at the open.
            if i > 0 and decisions[i - 1] == 1:
                if not holding:
                    if not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]):
                        continue
                    entry_price = opens[i] * (1.0 + params.entry_slippage_bps / 10000.0)
                    fixed_stop = entry_price - params.stop_atr * atr[i - 1]
                    holding = True
                    age = 0
                age += 1
                stopped = not np.isfinite(closes[i]) or closes[i] < fixed_stop
                if stopped or age >= 2:
                    holding = False
                    age = 0
                    decisions[i] = 0
                else:
                    decisions[i] = 1
            else:
                holding = False
                age = 0
                decisions[i] = int(setup[i])
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({
        "ordinal_entropy": 0.5, "autocorrelation": -0.5,
        "forecast_score": 0.5, "atr": 1.0,
        "complexity_agrees": True, "rebound_agrees": True,
    }, index=index)
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, 1, 1, 0, 1, 1]
    data.loc[index[1], "close"] = 98.0
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, 1, 0, 1, 1, 0]
    flat = data.copy()
    flat["close"] = 100.0
    derived = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, derived, None, params).data["signal"].eq(0).all()
