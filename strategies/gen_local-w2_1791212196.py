from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletMigrationParams:
    energy_window: int = 12
    migration_bars: int = 3
    migration_threshold: float = 0.035
    volume_window: int = 20
    pressure_window: int = 4
    pressure_threshold: float = 0.08
    profit_target: float = 0.015
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[WaveletMigrationParams]):
    strategy_id = "gen_local-w2_1791212196"

    @classmethod
    def params_type(cls):
        return WaveletMigrationParams

    def warmup_bars(self, params: WaveletMigrationParams) -> int:
        if min(params.energy_window, params.migration_bars,
               params.volume_window, params.pressure_window) < 1:
            raise ValueError("Windows must be positive.")
        if not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be between three and five.")
        if not all(np.isfinite(x) and x > 0 for x in (
            params.migration_threshold, params.pressure_threshold,
            params.profit_target)):
            raise ValueError("Thresholds must be finite and positive.")
        return int(max(8 + params.energy_window - 1 + params.migration_bars,
                       params.volume_window + params.pressure_window - 1))

    def indicators(self, data: pd.DataFrame,
                   params: WaveletMigrationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (1, 2, 4):
            out[f"haar_{half}"] = (
                out["log_close"].rolling(half, min_periods=half).sum()
                - out["log_close"].shift(half).rolling(half, min_periods=half).sum()
            ) / np.sqrt(2.0 * half)
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(
            axis=1, min_count=3
        )
        out["middle_share"] = out["energy_2"] / out["total_energy"].replace(0, np.nan)
        out["migration"] = out["middle_share"].diff(params.migration_bars)
        out["regime"] = np.select(
            [out["migration"] > params.migration_threshold,
             out["migration"] < -params.migration_threshold],
            [1, -1], default=0
        )
        out["volume_mean"] = data["volume"].where(data["volume"] >= 0).rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean()
        out["relative_volume"] = data["volume"].where(data["volume"] >= 0) / (
            out["volume_mean"].replace(0, np.nan)
        )
        out["body_fraction"] = (
            (data["close"] - data["open"])
            / (data["high"] - data["low"]).where(data["high"] > data["low"])
        ).clip(-1.0, 1.0)
        out["weighted_pressure"] = out["body_fraction"] * out["relative_volume"]
        out["pressure"] = (
            out["weighted_pressure"].rolling(params.pressure_window).sum()
            / out["relative_volume"].rolling(params.pressure_window).sum().replace(0, np.nan)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: WaveletMigrationParams) -> SignalFrame:
        warmup = self.warmup_bars(params)
        wavelet_vote = pd.Series(np.select(
            [indicators["regime"].eq(1), indicators["regime"].eq(-1)],
            [np.sign(indicators["haar_2"]), -np.sign(indicators["haar_1"])],
            default=0
        ), index=data.index).fillna(0)
        # Both primitives must agree; candle pressure never chooses the regime.
        candidate = wavelet_vote.where(
            wavelet_vote * indicators["pressure"] >= params.pressure_threshold, 0
        ).astype(int).to_numpy()
        candidate[:max(0, warmup - 1)] = 0
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if side:
                # Entry was decided yesterday; today's open is now observable.
                if i == entry_bar:
                    entry_price = opens[i]
                expired = i - entry_bar + 1 >= params.max_hold
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                gained = (
                    not invalid_entry and np.isfinite(closes[i])
                    and side * (closes[i] / entry_price - 1.0) >= params.profit_target
                )
                # Close-based targets exit at the following open.
                if invalid_entry or gained or expired:
                    side = 0
            elif candidate[i]:
                side = int(candidate[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletMigrationParams()
    w = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=w + 9, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    ctx = StrategyContext("SPY", "1d", w)
    votes = pd.DataFrame(
        {"regime": 1, "haar_1": 0.01, "haar_2": 0.01, "pressure": 0.2},
        index=index
    )
    votes.loc[index[:w], "pressure"] = 0.0
    signal = strategy.generate_signals(data, votes, ctx, params).data["signal"]
    assert signal.iloc[w] == 0
    assert signal.iloc[w + 1:w + 5].eq(1).all() and signal.iloc[w + 5] == 0
    for direction in (1, -1):
        trial = data.copy()
        trial.loc[index[w + 1], "close"] = 100.0 * (1 + direction * 0.02)
        agreed = votes.copy()
        agreed["haar_2"] *= direction
        agreed["pressure"] *= direction
        signal = strategy.generate_signals(trial, agreed, ctx, params).data["signal"]
        assert signal.iloc[w + 1] == direction and signal.iloc[w + 2] == 0
    opposed = votes.copy()
    opposed["pressure"] = -0.2
    assert strategy.generate_signals(data, opposed, ctx, params).data["signal"].eq(0).all()
    derived = strategy.indicators(data, params)
    assert strategy.generate_signals(data, derived, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
