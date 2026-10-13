from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 40
    tension_threshold: float = 1.5
    release_fraction: float = 0.08
    profit_target: float = 0.02
    max_hold: int = 8
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if self.energy_window < 2 or self.max_hold < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (self.tension_threshold > 0 and 0 < self.release_fraction < 1):
            raise ValueError("Invalid tension or release threshold.")
        if not (0 < self.profit_target < 1 and 0 < self.min_size <= self.max_size <= 1):
            raise ValueError("Invalid target or position sizes.")


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791338870"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 16 + params.energy_window + 1

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (2, 4, 8):
            block = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (block - block.shift(half))
        out["coarse_energy"] = out["haar_4"].pow(2) + out["haar_8"].pow(2)
        out["baseline_energy"] = out["coarse_energy"].rolling(
            params.energy_window, min_periods=params.energy_window
        ).mean().shift(1)
        out["tension"] = np.sqrt(
            out["coarse_energy"] / out["baseline_energy"].where(out["baseline_energy"] > 1e-16)
        )
        out["release"] = 1.0 - out["coarse_energy"] / out["coarse_energy"].shift(1).where(
            out["coarse_energy"].shift(1) > 1e-16
        )
        out["coherence"] = out["haar_4"] * out["haar_8"]
        out["opposition"] = out["haar_2"] * out["haar_8"]
        out["strength"] = ((out["tension"] - params.tension_threshold)
                           / params.tension_threshold).clip(0.0, 1.0)
        out["ready"] = out.notna().all(axis=1).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletParams
    ) -> SignalFrame:
        eligible = (
            indicators["ready"].eq(1)
            & indicators["tension"].ge(params.tension_threshold)
            & indicators["release"].ge(params.release_fraction)
            & indicators["coherence"].gt(0)
            & indicators["opposition"].lt(0)
        )
        candidates = pd.Series(
            np.where(eligible, -np.sign(indicators["haar_8"]), 0),
            index=data.index
        ).astype(int).to_numpy()
        proposed_sizes = (params.min_size + (params.max_size - params.min_size)
                          * indicators["strength"]).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        direction = 0
        entry_bar = -1
        entry_price = np.nan
        held_size = params.min_size
        for i in range(len(data)):
            if direction:
                if i == entry_bar:
                    # The prior close's decision fills at this bar's open.
                    entry_price = opens[i]
                valid = (np.isfinite(entry_price) and entry_price > 0
                         and np.isfinite(closes[i]) and closes[i] > 0)
                gain = direction * (closes[i] / entry_price - 1.0) if valid else 0.0
                if not valid or gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    direction = 0
                    # Keep this close flat; a later close may initiate a new trade.
                    continue
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                direction = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
                held_size = float(np.clip(proposed_sizes[i], params.min_size, params.max_size))
            signals[i] = direction
            sizes[i] = held_size if direction else params.min_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletParams(max_hold=3)
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0},
        index=index
    )
    ind = pd.DataFrame(
        {"ready": 1, "tension": 3.0, "release": 0.2, "coherence": 1.0,
         "opposition": -1.0, "haar_8": -1.0, "strength": 1.0}, index=index
    )
    ind.loc[index[1:], "ready"] = 0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert result["size"].iloc[1] == params.max_size
    data.loc[index[1], "close"] = 103.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind["haar_8"] = 1.0
    data.loc[index[1], "close"] = 97.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert (result["size"] > 0).all()
    flat = strategy.indicators(data.assign(close=100.0), params)
    assert np.isfinite(flat.to_numpy()).all()
    assert strategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
