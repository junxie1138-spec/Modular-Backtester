from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    coherence_threshold: float = 0.35
    profit_target: float = 0.012


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791201214"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None)
        acceleration = returns.diff()
        positive = acceleration.clip(lower=0)
        positive_mass = positive.rolling(4, min_periods=4).sum()
        largest_impulse = positive.rolling(4, min_periods=4).max()
        gross = acceleration.abs().rolling(4, min_periods=4).sum()
        coherence = acceleration.rolling(4, min_periods=4).sum() / gross.where(gross > 0)
        volatility = returns.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        exposure = (0.12 / volatility.where(volatility > 0)).clip(lower=0.05, upper=0.95)
        entry = (
            (coherence > params.coherence_threshold)
            & (positive_mass > 2.0 * largest_impulse)
            & (acceleration > 0)
            & (returns > 0)
            & (volatility > 0)
            & np.isfinite(volatility)
        )
        return pd.DataFrame({
            "returns": returns,
            "acceleration": acceleration,
            "positive_mass": positive_mass,
            "largest_impulse": largest_impulse,
            "gross_acceleration": gross,
            "coherence": coherence,
            "volatility": volatility,
            "exposure": exposure,
            "entry": entry,
        }, index=data.index).replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if not (0.0 < params.coherence_threshold < 1.0):
            raise ValueError("coherence_threshold must be between zero and one")
        if not (np.isfinite(params.profit_target) and params.profit_target > 0.0):
            raise ValueError("profit_target must be positive and finite")
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.95, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        exposures = indicators["exposure"].to_numpy(dtype=float)
        holding = False
        entry_bar = -1
        entry_price = 0.0
        position_size = 0.95
        for i in range(len(data)):
            if holding:
                # The entry decision was yesterday; today's open is now observed.
                if i == entry_bar:
                    entry_price = opens[i]
                valid = (
                    np.isfinite(entry_price) and entry_price > 0.0
                    and np.isfinite(closes[i]) and closes[i] > 0.0
                )
                if (
                    not valid
                    or closes[i] >= entry_price * (1.0 + params.profit_target)
                    or i - entry_bar + 1 >= 4
                ):
                    holding = False
            elif (
                entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0
                and np.isfinite(exposures[i]) and exposures[i] > 0.0
            ):
                holding = True
                entry_bar = i + 1
                position_size = float(np.clip(exposures[i], 0.05, 0.95))
            if holding:
                signals[i] = 1
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, False, False, False, False],
        "exposure": 0.4,
    }, index=index)
    stopped = strategy.generate_signals(data, indicators, None, params).data
    assert stopped["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert np.allclose(stopped.loc[stopped["signal"] == 1, "size"], 0.4)
    data.loc[index[2], "close"] = 102.0
    targeted = strategy.generate_signals(data, indicators, None, params).data
    assert targeted["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert not derived.isna().any().any()
    assert not derived["entry"].any()


if __name__ == "__main__":
    _self_check()
