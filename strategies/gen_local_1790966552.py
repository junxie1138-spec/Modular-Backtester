from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class LatentSlopeParams:
    noise_window: int = 40
    level_gain: float = 0.20
    slope_gain: float = 0.02
    innovation_gate: float = 0.35
    profit_target: float = 0.04
    max_hold_bars: int = 18

    def __post_init__(self):
        if self.noise_window < 2 or self.max_hold_bars < 1:
            raise ValueError("Window and holding period must be positive.")
        if not 0 < self.slope_gain < self.level_gain < 1:
            raise ValueError("Require 0 < slope_gain < level_gain < 1.")
        if not np.isfinite(self.innovation_gate) or self.innovation_gate <= 0:
            raise ValueError("innovation_gate must be finite and positive.")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive.")


class GeneratedStrategy(BaseStrategy[LatentSlopeParams]):
    strategy_id = "gen_local_1790966552"

    @classmethod
    def params_type(cls):
        return LatentSlopeParams

    def warmup_bars(self, params: LatentSlopeParams) -> int:
        # Recursive filter burn-in, plus the differenced volatility window.
        return int(max(params.noise_window + 1,
                       np.ceil(10 / params.level_gain),
                       np.ceil(4 / np.sqrt(params.slope_gain))))

    def indicators(self, data: pd.DataFrame, params: LatentSlopeParams) -> pd.DataFrame:
        price = np.log(data["close"].where(data["close"] > 0))
        noise = price.diff().rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0).replace(0, np.nan)
        n = len(data)
        level = np.full(n, np.nan)
        slope = np.full(n, np.nan)
        innovation = np.full(n, np.nan)
        slope_innovation = np.full(n, np.nan)
        last_level, last_slope = np.nan, 0.0
        # Fixed-gain latent-state updates are recursive rather than rolling.
        for i, observed in enumerate(price.to_numpy(dtype=float)):
            if not np.isfinite(observed):
                last_level, last_slope = np.nan, 0.0
                continue
            if not np.isfinite(last_level):
                last_level = observed
                level[i], slope[i] = last_level, last_slope
                continue
            error = observed - (last_level + last_slope)
            correction = params.slope_gain * error
            last_level = last_level + last_slope + params.level_gain * error
            last_slope += correction
            level[i], slope[i] = last_level, last_slope
            innovation[i], slope_innovation[i] = error, correction
        out = pd.DataFrame({
            "log_price": price, "noise": noise,
            "latent_level": level, "latent_slope": slope,
            "innovation": innovation, "slope_innovation": slope_innovation,
        }, index=data.index)
        out["slope_innovation_z"] = out["slope_innovation"] / (
            params.slope_gain * out["noise"]
        )
        out["rising_regime"] = out["latent_slope"].gt(0)
        # Falling latent trends need twice the slope correction to qualify.
        out["entry_gate"] = np.where(
            out["rising_regime"], params.innovation_gate,
            2.0 * params.innovation_gate
        )
        out["ready"] = price.notna().rolling(
            self.warmup_bars(params), min_periods=self.warmup_bars(params)
        ).sum().eq(self.warmup_bars(params))
        out["qualifies"] = (
            out["ready"] & out["slope_innovation_z"].gt(out["entry_gate"])
            & np.isfinite(out["slope_innovation_z"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: LatentSlopeParams) -> SignalFrame:
        qualified = indicators["qualifies"].fillna(False).astype(bool)
        confirmed = qualified & qualified.shift(1, fill_value=False)
        fresh = confirmed & ~confirmed.shift(1, fill_value=False)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active, entry_bar, entry_price = False, -1, np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Today's open is known now; never read tomorrow's open.
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                profit_exit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1 + params.profit_target)
                )
                time_exit = i - entry_bar + 1 >= params.max_hold_bars
                if profit_exit or time_exit:
                    active = False
                    continue
            elif fresh.iloc[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active, entry_bar, entry_price = True, i + 1, np.nan
            decisions[i] = int(active)
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = LatentSlopeParams(max_hold_bars=3)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({"qualifies": True}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    signal = strategy.generate_signals(data, indicators, None, params).data
    assert signal["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    data.loc[index[3], "close"] = 105.0
    signal = strategy.generate_signals(data, indicators, None, params).data
    assert signal["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    indicators["qualifies"] = [True, False, True, False, True, False, True, False]
    assert not strategy.generate_signals(data, indicators, None, params).data["signal"].any()
    assert strategy.indicators(data, params)["qualifies"].eq(False).all()


if __name__ == "__main__":
    _self_check()
