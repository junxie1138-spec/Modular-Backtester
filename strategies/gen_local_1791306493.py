from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    excursion_window: int = 20
    energy_window: int = 6
    atr_window: int = 14
    energy_ratio: float = 1.3
    excursion_atr: float = 1.5
    trail_atr: float = 1.8
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791306493"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.excursion_window, params.atr_window + 1,
                       params.energy_window + 8))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.excursion_window, params.energy_window, params.atr_window) < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if min(params.energy_ratio, params.excursion_atr, params.trail_atr) <= 0:
            raise ValueError("Thresholds must be positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two bars")
        close = data["close"].astype(float)
        previous = close.shift(1)
        returns = np.log(close.where(close > 0)).diff()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["returns"] = returns
        out["atr"] = true_range.rolling(params.atr_window).mean().where(lambda x: x > 0)
        for half in (1, 2, 4):
            # Causal Haar detail: two adjacent blocks, ending at this close.
            recent = returns.rolling(half).mean()
            detail = (recent - recent.shift(half)) * np.sqrt(half / 2.0)
            out[f"detail_{2 * half}"] = detail
            out[f"energy_{2 * half}"] = detail.pow(2).rolling(params.energy_window).mean()
        out["drawdown_atr"] = (close.rolling(params.excursion_window).max() - close) / out["atr"]
        out["runup_atr"] = (close - close.rolling(params.excursion_window).min()) / out["atr"]
        out["fine_falling"] = out["energy_2"] < out["energy_2"].shift(1)
        out["coarse_dominant"] = out["energy_8"] > params.energy_ratio * (out["energy_2"] + out["energy_4"])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        valid = indicators[["returns", "atr", "detail_8", "energy_2", "energy_4", "energy_8",
                            "drawdown_atr", "runup_atr"]].notna().all(axis=1)
        migration = valid & indicators["fine_falling"].fillna(False) & indicators["coarse_dominant"].fillna(False)
        long_setup = (migration & (indicators["drawdown_atr"] >= params.excursion_atr)
                      & (indicators["detail_8"] < 0) & (indicators["returns"] > 0))
        short_setup = (migration & (indicators["runup_atr"] >= params.excursion_atr)
                       & (indicators["detail_8"] > 0) & (indicators["returns"] < 0))
        long_entry = (long_setup & long_setup.shift(1, fill_value=False)).to_numpy()
        short_entry = (short_setup & short_setup.shift(1, fill_value=False)).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = -np.inf
        low_water = np.inf
        trailing_floor = -np.inf
        trailing_ceiling = np.inf
        for i in range(len(data)):
            if position:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                    position = 0
                else:
                    # Start tracking on the fill bar; never loosen either stop.
                    high_water = max(high_water, closes[i])
                    low_water = min(low_water, closes[i])
                    trailing_floor = max(trailing_floor, high_water - params.trail_atr * atrs[i])
                    trailing_ceiling = min(trailing_ceiling, low_water + params.trail_atr * atrs[i])
                    high_stop = closes[i] < trailing_floor
                    short_stop = position == -1 and closes[i] > trailing_ceiling
                    if high_stop or short_stop or age >= params.max_hold_bars:
                        position = 0
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                age = 0
                high_water = -np.inf
                low_water = np.inf
                trailing_floor = -np.inf
                trailing_ceiling = np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100.0, 100.0, 102.0, 103.0, 104.0, 105.0]}, index=index)
    ind = pd.DataFrame({
        "returns": 0.01, "atr": 1.0, "detail_8": -0.02,
        "energy_2": 0.001, "energy_4": 0.001, "energy_8": 0.01,
        "drawdown_atr": 2.0, "runup_atr": 0.0,
        "fine_falling": [True, True, False, False, False, False],
        "coarse_dominant": True,
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, ind, None, Params())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    falling = data.copy()
    falling.loc[index[3], "close"] = 99.0
    result = GeneratedStrategy().generate_signals(falling, ind, None, Params())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 0]
