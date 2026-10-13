from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.45
    fractional_window: int = 32
    normalization_window: int = 12
    regime_window: int = 24
    pressure_window: int = 8
    regime_enter: float = 0.55
    regime_leave: float = 0.25
    entry_threshold: float = 0.25
    pressure_threshold: float = 0.05


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790897911"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.fractional_window + max(params.normalization_window, params.regime_window, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if min(params.fractional_window, params.normalization_window, params.regime_window, params.pressure_window) < 2:
            raise ValueError("All windows must be at least two")
        if not -1.0 <= params.regime_leave < params.regime_enter <= 1.0:
            raise ValueError("Regime thresholds must be ordered within [-1, 1]")
        if params.entry_threshold < 0.0 or not 0.0 <= params.pressure_threshold < 1.0:
            raise ValueError("Invalid entry thresholds")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional = np.full(len(data), np.nan)
        if len(data) >= params.fractional_window:
            fractional[params.fractional_window - 1:] = np.convolve(
                out["log_close"].to_numpy(), weights, mode="valid"
            )
        out["fractional_close"] = fractional
        out["fractional_mean"] = out["fractional_close"].rolling(params.normalization_window).mean()
        out["fractional_std"] = out["fractional_close"].rolling(params.normalization_window).std(ddof=0)
        out["fractional_z"] = (
            (out["fractional_close"] - out["fractional_mean"])
            / out["fractional_std"].where(out["fractional_std"] > 1e-12)
        )
        out["fractional_lag"] = out["fractional_close"].shift(1)
        out["serial_dependence"] = out["fractional_close"].rolling(params.regime_window).corr(out["fractional_lag"])
        span = data["high"] - data["low"]
        out["close_location"] = (
            (2.0 * data["close"] - data["high"] - data["low"])
            / span.where(span > 0.0)
        ).where(span != 0.0, 0.0).clip(-1.0, 1.0)
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0.0)
        out["signed_volume"] = out["close_location"] * out["valid_volume"]
        out["flow_sum"] = out["signed_volume"].rolling(params.pressure_window).sum()
        out["volume_sum"] = out["valid_volume"].rolling(params.pressure_window).sum()
        out["money_flow"] = out["flow_sum"] / out["volume_sum"].where(out["volume_sum"] > 0.0)
        switches = pd.Series(np.nan, index=data.index, dtype=float)
        switches.loc[out["serial_dependence"] >= params.regime_enter] = 1.0
        switches.loc[out["serial_dependence"] <= params.regime_leave] = 0.0
        out["continuation_regime"] = switches.ffill().fillna(0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        persistent = indicators["continuation_regime"].eq(1.0)
        fractional_agrees = (
            (persistent & indicators["fractional_z"].gt(params.entry_threshold))
            | (~persistent & indicators["fractional_z"].lt(-params.entry_threshold))
        )
        pressure_agrees = indicators["money_flow"].gt(params.pressure_threshold)
        ready = indicators[["fractional_z", "serial_dependence", "money_flow"]].notna().all(axis=1)
        df = pd.DataFrame(index=data.index)
        # The AND condition defines the held position; its reversal is the only exit.
        df["signal"] = (ready & fractional_agrees & pressure_agrees).astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    n = 180
    t = np.arange(n, dtype=float)
    close = 100.0 * np.exp(0.001 * t + 0.02 * np.sin(t / 7.0))
    data = pd.DataFrame(
        {"open": close - 0.3, "high": close + 0.5, "low": close - 1.0,
         "close": close, "volume": np.full(n, 1000000.0)},
        index=pd.date_range("2020-01-01", periods=n),
    )
    strategy, params = GeneratedStrategy(), Params()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    derived = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, derived, ctx, params).data
    cut = 130
    prefix = data.iloc[:cut]
    prefix_signals = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(signals.iloc[:cut], prefix_signals)
    assert set(signals["signal"].unique()) == {0, 1}
    assert (signals["size"] > 0.0).all()
    manual = derived.copy()
    manual["serial_dependence"] = 0.8
    manual["continuation_regime"] = 1.0
    manual["fractional_z"] = 1.0
    manual["money_flow"] = 0.2
    manual.iloc[3, manual.columns.get_loc("money_flow")] = -0.2
    shifted = strategy.generate_signals(data, manual, ctx, params).data["signal"]
    assert shifted.iloc[:6].tolist() == [0, 1, 1, 1, 0, 1]
