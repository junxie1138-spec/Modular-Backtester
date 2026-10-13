from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_window: int = 8
    vol_window: int = 20
    atr_window: int = 14
    innovation_threshold: float = 1.0
    stop_atr: float = 1.5
    target_daily_vol: float = 0.008


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790921815"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.trend_window + 2, params.vol_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.trend_window, params.vol_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if min(params.innovation_threshold, params.stop_atr, params.target_daily_vol) <= 0:
            raise ValueError("Threshold, stop and volatility target must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        previous = close.shift(1)
        out["log_return"] = np.log(close / previous)
        # A finite-window local drift estimates the latent log-price slope.
        out["latent_slope"] = out["log_return"].rolling(params.trend_window).mean()
        out["daily_vol"] = out["log_return"].rolling(params.vol_window).std(ddof=0)
        out["innovation"] = (
            (out["log_return"] - out["latent_slope"].shift(1))
            / out["daily_vol"].shift(1).where(out["daily_vol"].shift(1) > 0)
        )
        out["overnight"] = np.log(opening / previous)
        out["intraday"] = np.log(close / opening)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["strength"] = (out["innovation"].abs() / 3.0).clip(0.0, 1.0)
        out["exposure"] = (
            params.target_daily_vol / out["daily_vol"].where(out["daily_vol"] > 0)
            * out["strength"]
        ).clip(0.05, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        required = ["innovation", "overnight", "intraday", "atr", "exposure"]
        valid = np.isfinite(indicators[required]).all(axis=1)
        shock = indicators["innovation"]
        eligible = (
            valid & (indicators["atr"] > 0)
            & (shock.abs() >= params.innovation_threshold)
            & (indicators["overnight"] * shock < 0)
            & (indicators["intraday"] * shock > 0)
        )
        candidates = np.where(eligible, np.sign(shock.fillna(0)), 0).astype(int)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        exposure = indicators["exposure"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        side, decision_bar = 0, -1
        entry_price, entry_atr, held_size = np.nan, np.nan, 1.0

        for i in range(len(data)):
            if side:
                # Entry reference is the next bar's open, observed only on that bar.
                if i == decision_bar + 1:
                    entry_price = opening[i]
                invalid = not (np.isfinite(entry_price) and np.isfinite(closing[i]))
                stopped = (
                    not invalid
                    and side * (closing[i] - entry_price) < -params.stop_atr * entry_atr
                )
                if invalid or stopped or i - decision_bar >= 2:
                    side = 0
                    continue
                signals[i], sizes[i] = side, held_size
            elif candidates[i]:
                side = int(candidates[i])
                decision_bar = i
                entry_atr = atr[i]
                entry_price = np.nan
                held_size = float(exposure[i])
                signals[i], sizes[i] = side, held_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    features = pd.DataFrame(
        {"innovation": 0.0, "overnight": 0.0, "intraday": 0.0,
         "atr": 1.0, "exposure": 0.4}, index=index,
    )
    for bar, direction in [(1, 1), (5, -1), (8, 1)]:
        features.iloc[bar] = [2.0 * direction, -0.01 * direction,
                              0.02 * direction, 1.0, 0.4]
    data.loc[index[2], "close"] = 98.0
    data.loc[index[6], "close"] = 102.0
    strategy = GeneratedStrategy()
    params = StrategyParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 1, 0]
    assert result.loc[index[2], "size"] == 0.4
    assert np.isfinite(result["size"]).all() and (result["size"] > 0).all()
    prefix = strategy.generate_signals(data.iloc[:7], features.iloc[:7], ctx, params).data
    pd.testing.assert_frame_equal(prefix, result.iloc[:7])


if __name__ == "__main__":
    _self_check()
