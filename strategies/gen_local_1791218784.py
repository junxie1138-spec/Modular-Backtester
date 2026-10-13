from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 12
    coarse_share: float = 0.70
    pressure_window: int = 16
    pressure_threshold: float = 0.12
    profit_target: float = 0.045
    max_hold_bars: int = 18

    def __post_init__(self):
        if self.energy_window < 1 or self.pressure_window < 1 or self.max_hold_bars < 1:
            raise ValueError("Window lengths and holding period must be positive.")
        if not 0.0 < self.coarse_share < 1.0:
            raise ValueError("coarse_share must be between zero and one.")
        if not 0.0 <= self.pressure_threshold < 1.0:
            raise ValueError("pressure_threshold must be in [0, 1).")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive.")


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791218784"

    @classmethod
    def params_type(cls):
        return WaveletParams

    @staticmethod
    def warmup_bars(params):
        return int(max(32 + params.energy_window - 1, params.pressure_window))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        for scale in (2, 4, 8, 16):
            block = out["log_close"].rolling(scale, min_periods=scale).mean()
            out[f"block_{scale}"] = block
            # Causal, undecimated Haar detail: recent block minus adjacent older block.
            out[f"haar_{scale}"] = (block - block.shift(scale)) * np.sqrt(scale / 2.0)
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out[[f"energy_{s}" for s in (2, 4, 8, 16)]].sum(
            axis=1, min_count=4
        )
        out["coarse_energy"] = out["energy_8"] + out["energy_16"]
        out["coarse_share"] = out["coarse_energy"] / out["total_energy"].where(
            out["total_energy"] > 0.0
        )
        out["range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["close_location"] = (
            (2.0 * data["close"] - data["high"] - data["low"])
            / out["range"].where(out["range"] > 0.0)
        ).clip(-1.0, 1.0)
        out.loc[out["range"].eq(0.0), "close_location"] = 0.0
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0.0)
        out["weighted_pressure"] = out["close_location"] * out["valid_volume"]
        out["pressure_sum"] = out["weighted_pressure"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["volume_sum"] = out["valid_volume"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["pressure"] = out["pressure_sum"] / out["volume_sum"].where(
            out["volume_sum"] > 0.0
        )
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        strong = indicators["coarse_share"].ge(params.coarse_share)
        long_entry = (
            strong & indicators["haar_8"].gt(0.0) & indicators["haar_16"].gt(0.0)
            & indicators["pressure"].gt(params.pressure_threshold)
        )
        short_entry = (
            strong & indicators["haar_8"].lt(0.0) & indicators["haar_16"].lt(0.0)
            & indicators["pressure"].lt(-params.pressure_threshold)
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if side:
                # Read the entry open only when that bar arrives, never at decision time.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        side = 0
                        continue
                held_bars = i - entry_bar + 1
                gain = side * (closes[i] / entry_price - 1.0)
                if held_bars >= params.max_hold_bars or (
                    np.isfinite(gain) and gain >= params.profit_target
                ):
                    side = 0
                    continue
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                side = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    params = WaveletParams(max_hold_bars=3)
    for direction in (1, -1):
        ind = pd.DataFrame(
            {"coarse_share": 0.9, "haar_8": float(direction),
             "haar_16": float(direction), "pressure": 0.5 * direction},
            index=index,
        )
        result = GeneratedStrategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, direction, direction, direction]
        changed = data.copy()
        changed.loc[index[2], "close"] = 106.0 if direction == 1 else 94.0
        result = GeneratedStrategy.generate_signals(changed, ind, None, params).data
        assert result["signal"].iloc[:4].tolist() == [0, direction, direction, 0]
        prefix = GeneratedStrategy.generate_signals(changed.iloc[:3], ind.iloc[:3], None, params).data
        assert prefix["signal"].equals(result["signal"].iloc[:3])
    ind = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].eq(0).all()
    assert result["size"].gt(0.0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 32 + params.energy_window - 1


if __name__ == "__main__":
    _self_check()
