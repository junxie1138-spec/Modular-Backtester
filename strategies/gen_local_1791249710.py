from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    energy_window: int = 10
    coarse_share: float = 0.60
    profit_target: float = 0.04
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791249710"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, 16 + max(1, int(params.energy_window)))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        window = max(1, int(params.energy_window))
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        out["gap"] = np.log(opening / previous)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["gap_bias"] = out["gap"].rolling(8, min_periods=8).mean()
        # Causal Haar details: adjacent half-window sums, normalized to unit norm.
        for scale in (2, 4, 8, 16):
            half = scale // 2
            recent = out["gap"].rolling(half, min_periods=half).mean()
            detail = np.sqrt(half / 2.0) * (recent - recent.shift(half))
            out[f"haar{scale}"] = detail
            out[f"energy{scale}"] = detail.pow(2).rolling(window, min_periods=window).mean()
        out["coarse_energy"] = out["energy8"] + out["energy16"]
        out["total_energy"] = out["coarse_energy"] + out["energy2"] + out["energy4"]
        out["coarse_fraction"] = out["coarse_energy"] / out["total_energy"].where(out["total_energy"] > 1e-12)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        onset = (
            ready
            & (out["coarse_fraction"] > params.coarse_share)
            & (out["coarse_fraction"].shift(1) <= params.coarse_share)
        )
        long_entry = onset & (out["gap_bias"] > 0) & (out["haar16"] > 0) & (data["close"] > out["ma200"])
        short_entry = onset & (out["gap_bias"] < 0) & (out["haar16"] < 0) & (data["close"] < out["ma200"])
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        target = max(0.000001, float(params.profit_target))
        duration = max(1, int(params.max_hold))
        raw = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                # Yesterday's decision fills at today's open; never inspect a future bar.
                if entry_bar < 0:
                    entry_price = opens[i]
                    entry_bar = i
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                if not valid_entry or gain >= target or i - entry_bar + 1 >= duration:
                    position = 0
                    entry_price = np.nan
                    entry_bar = -1
            elif i >= self.warmup_bars(params) - 1 and entries[i] in (-1, 1):
                if np.isfinite(closes[i]) and closes[i] > 0:
                    position = int(entries[i])
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 94.0, "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    entries = pd.DataFrame({"entry": np.zeros(len(data), dtype=int)}, index=index)
    for direction in (1, -1):
        entries.iloc[200, 0] = direction
        data.iloc[202, data.columns.get_loc("close")] = 100.0 * (1.0 + direction * 0.05)
        result = strategy.generate_signals(data, entries, None, Params()).data
        assert result["signal"].iloc[200:204].tolist() == [0, direction, direction, 0]
        data["close"] = 100.0
        result = strategy.generate_signals(data, entries, None, Params(max_hold=3)).data
        assert result["signal"].iloc[200:205].tolist() == [0, direction, direction, direction, 0]
        assert result["size"].gt(0).all()
    original = strategy.indicators(data, Params())
    changed = data.copy()
    changed.iloc[220:, changed.columns.get_loc("open")] = 105.0
    pd.testing.assert_frame_equal(original.iloc[:220], strategy.indicators(changed, Params()).iloc[:220])
    assert not original["entry"].any()


if __name__ == "__main__":
    _self_check()
