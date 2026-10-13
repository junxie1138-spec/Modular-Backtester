from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EpidemicParams:
    frac_order: float = 0.45
    frac_window: int = 32
    center_window: int = 40
    susceptible_window: int = 20
    incidence_window: int = 4
    reproduction_min: float = 1.5
    vol_window: int = 20
    target_vol: float = 0.12
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[EpidemicParams]):
    strategy_id = "gen_local-w2_1791242453"

    @classmethod
    def params_type(cls):
        return EpidemicParams

    def warmup_bars(self, params: EpidemicParams) -> int:
        return int(max(params.frac_window + params.center_window +
                       params.susceptible_window + 2 * params.incidence_window + 2,
                       params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: EpidemicParams) -> pd.DataFrame:
        if not (0 < params.frac_order < 1 and params.reproduction_min > 1 and
                params.target_vol > 0 and 3 <= params.holding_bars <= 5):
            raise ValueError("Invalid fractional order, reproduction, volatility target, or holding period")
        if min(params.frac_window, params.center_window, params.susceptible_window,
               params.incidence_window, params.vol_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        log_close = np.log(close.where(close > 0))
        weights = np.ones(params.frac_window, dtype=float)
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        # ponytail: finite fractional filter; increase frac_window for longer memory.
        frac = log_close.copy()
        for k in range(1, params.frac_window):
            frac = frac + weights[k] * log_close.shift(k)
        center = frac.rolling(params.center_window, min_periods=params.center_window).mean()
        displacement = frac - center
        positive = displacement.gt(0).astype(float).where(displacement.notna())
        incidence = positive.rolling(params.incidence_window, min_periods=params.incidence_window).mean()
        prior_incidence = incidence.shift(params.incidence_window)
        susceptible = (1 - positive).shift(params.incidence_window).rolling(
            params.susceptible_window, min_periods=params.susceptible_window).mean()
        reproduction = incidence / prior_incidence.clip(lower=1.0 / params.incidence_window)
        returns = log_close.diff()
        volatility = returns.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        size = (params.target_vol / volatility.where(volatility > 0)).clip(0.05, 0.95)
        eligible = ((incidence >= 0.75) & (susceptible >= 0.55) &
                    (reproduction >= params.reproduction_min) & size.notna())
        return pd.DataFrame({
            "fractional_close": frac, "fractional_center": center,
            "displacement": displacement, "positive_state": positive,
            "incidence": incidence, "prior_incidence": prior_incidence,
            "susceptible": susceptible, "reproduction": reproduction,
            "log_return": returns, "annual_volatility": volatility,
            "target_size": size.fillna(0.05), "eligible": eligible.fillna(False),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: EpidemicParams) -> SignalFrame:
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five")
        eligible = indicators["eligible"].fillna(False).astype(bool)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy()
        targets = indicators["target_size"].replace([np.inf, -np.inf], np.nan).fillna(0.05).clip(0.05, 0.95).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.05, dtype=float)
        exit_at = -1
        entry_size = 0.05
        for i in range(len(data)):
            if exit_at >= 0 and i >= exit_at:
                exit_at = -1
                continue
            if exit_at < 0 and confirmed[i]:
                exit_at = i + params.holding_bars
                entry_size = float(targets[i])
            if exit_at >= 0:
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "eligible": [True, False, True, True] + [False] * 8,
        "target_size": np.linspace(0.1, 0.9, 12),
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, EpidemicParams()).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert np.allclose(result["size"].iloc[4:8], indicators["target_size"].iloc[3])
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
