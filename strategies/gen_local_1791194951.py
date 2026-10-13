from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    atr_window: int = 14
    regime_window: int = 40
    gap_min: float = 0.4
    elastic_max: float = 0.3
    plastic_min: float = 0.9
    stop_atr: float = 1.2
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local_1791194951"

    @classmethod
    def params_type(cls):
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        return int(params.atr_window + params.regime_window + 2)

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        scale = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["gap_atr"] = (data["open"] - prior_close) / scale
        out["response_atr"] = (data["close"] - prior_close) / scale
        out["gap_energy"] = out["gap_atr"].pow(2).rolling(
            params.regime_window, min_periods=params.regime_window
        ).sum().shift(1)
        out["gap_response"] = (out["gap_atr"] * out["response_atr"]).rolling(
            params.regime_window, min_periods=params.regime_window
        ).sum().shift(1)
        out["retention"] = out["gap_response"] / out["gap_energy"].where(out["gap_energy"] > 1e-12)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapParams
    ) -> SignalFrame:
        if (params.stop_atr <= 0 or params.gap_min <= 0
                or params.hold_bars not in (1, 2)
                or params.elastic_max >= params.plastic_min):
            raise ValueError("Invalid gap, stop, regime, or holding parameters")
        gap = indicators["gap_atr"].to_numpy(dtype=float)
        retention = indicators["retention"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        elastic = retention < params.elastic_max
        plastic = retention > params.plastic_min
        valid = (np.isfinite(gap) & np.isfinite(retention)
                 & np.isfinite(atr) & (atr > 0)
                 & np.isfinite(opens) & np.isfinite(closes)
                 & (opens > 0) & (closes > 0))
        direction = np.where(elastic, -np.sign(gap), np.where(plastic, np.sign(gap), 0))
        candidates = np.where(valid & (np.abs(gap) >= params.gap_min), direction, 0).astype(int)
        margin = np.where(elastic, params.elastic_max - retention, retention - params.plastic_min)
        conviction = np.clip(
            0.2 + 0.4 * (np.abs(gap) - params.gap_min) + 0.4 * np.maximum(margin, 0),
            0.2, 1.0
        )
        conviction = np.nan_to_num(conviction, nan=0.2, posinf=1.0, neginf=0.2)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        position, age = 0, 0
        entry, fixed_atr, position_size = 0.0, 0.0, 1.0
        for i in range(len(data)):
            if position:
                # The previous close's decision fills at this bar's open.
                if age == 0:
                    entry = opens[i]
                age += 1
                invalid_price = not (np.isfinite(entry) and entry > 0
                                     and np.isfinite(closes[i]) and closes[i] > 0)
                stopped = position * (closes[i] - entry) < -params.stop_atr * fixed_atr
                if invalid_price or stopped or age >= params.hold_bars:
                    position = 0
                    # Keep the exit decision flat; do not re-enter on this close.
                    continue
            elif candidates[i]:
                position = int(candidates[i])
                age = 0
                fixed_atr = atr[i]
                position_size = float(conviction[i])
            signals[i] = position
            sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100., 110., 100., 100., 100., 100., 100., 100.],
        "high": [111.] * 8, "low": [98.] * 8,
        "close": [100., 108., 100., 100., 100., 100., 100., 100.],
        "volume": [1000.] * 8,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.] * 8,
        "gap_atr": [1., 1., 0., -1., 0., 0., 0., 0.],
        "retention": [1.2] * 8,
    }, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, GapParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0, 0]
    assert 0 < result["size"].iloc[1] < 1
    data.loc[index[4], "close"] = 102.
    result = strategy.generate_signals(data, indicators, None, GapParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
