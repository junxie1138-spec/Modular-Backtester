from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    baseline_bars: int = 60
    energy_trigger: float = 1.6
    fine_energy_cap: float = 1.3
    profit_target: float = 0.04
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791315999"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 16 + 4 + params.baseline_bars

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        price = np.log(data["close"].where(data["close"] > 0))
        out["log_close"] = price
        for name, half in (("fine", 2), ("medium", 4), ("coarse", 8)):
            average = price.rolling(half, min_periods=half).mean()
            coefficient = (average - average.shift(half)) * np.sqrt(half / 2.0)
            energy = coefficient.pow(2).rolling(4, min_periods=4).mean()
            baseline = energy.shift(4).rolling(
                params.baseline_bars, min_periods=params.baseline_bars
            ).mean()
            out[name + "_haar"] = coefficient
            out[name + "_energy"] = energy
            out[name + "_baseline"] = baseline
            out[name + "_ratio"] = (energy / baseline.where(baseline > 0)).replace(
                [np.inf, -np.inf], np.nan
            )
        out["candidate"] = (
            (out["coarse_ratio"] >= params.energy_trigger)
            & (out["fine_ratio"] <= params.fine_energy_cap)
            & (out["coarse_haar"] > 0)
            & (out["medium_haar"] > 0)
        ).fillna(False)
        out["confirmed"] = (
            out["candidate"] & out["candidate"].shift(1, fill_value=False)
        )
        out["rearm"] = (
            (out["coarse_ratio"] < 1.0) | (out["coarse_haar"] <= 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        rearm = indicators["rearm"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        armed = True
        active = False
        pending = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            # Record the entry only when its next-bar execution open is available.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i
            if active:
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.holding_bars
                if target_hit or expired:
                    active = False
                else:
                    decisions[i] = 1
                continue
            if rearm[i]:
                armed = True
            if armed and confirmed[i]:
                decisions[i] = 1
                pending = True
                armed = False
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    candidate = pd.Series(True, index=index)
    indicators = pd.DataFrame(
        {"confirmed": candidate & candidate.shift(1, fill_value=False),
         "rearm": False}, index=index
    )
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(
        data, indicators, None, WaveletParams(holding_bars=2)
    ).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0]
    data.loc[index[2], "close"] = 105.0
    result = strategy.generate_signals(data, indicators, None, WaveletParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0, 0]
    flat = data.copy()
    flat["close"] = 100.0
    derived = strategy.indicators(flat, WaveletParams())
    assert not derived["confirmed"].any()
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
