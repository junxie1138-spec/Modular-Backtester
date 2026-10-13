from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeSeasonParams:
    range_window: int = 20
    season_window: int = 126
    atr_window: int = 14
    min_strength: float = 0.08
    stop_atr: float = 1.2
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeSeasonParams]):
    strategy_id = "gen_local_1791295031"

    @classmethod
    def params_type(cls) -> type[RangeSeasonParams]:
        return RangeSeasonParams

    @staticmethod
    def warmup_bars(params: RangeSeasonParams) -> int:
        return int(max(params.range_window, params.atr_window) + params.season_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeSeasonParams) -> pd.DataFrame:
        if min(params.range_window, params.atr_window) < 2 or params.season_window < 20:
            raise ValueError("Indicator windows are too short")
        if params.min_strength <= 0 or params.stop_atr <= 0 or params.holding_bars not in (1, 2):
            raise ValueError("Invalid strength, stop, or holding parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["range_baseline"] = out["range"].rolling(params.range_window).mean().shift(1)
        out["range_shock"] = (out["range"] / out["range_baseline"].replace(0, np.nan) - 1).clip(lower=0)
        out["close_location"] = ((2 * data["close"] - data["high"] - data["low"]) / out["range"].replace(0, np.nan)).clip(-1, 1).fillna(0)
        out["pressure"] = out["range_shock"] * out["close_location"]
        out["true_range"] = pd.concat([out["range"], (data["high"] - out["previous_close"]).abs(), (data["low"] - out["previous_close"]).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["atr_fraction"] = out["atr"] / data["close"].where(data["close"] > 0)
        out["intraday_return"] = data["close"] / data["open"].where(data["open"] > 0) - 1
        out["weekday"] = data.index.dayofweek
        out["prior_weekday"] = out["weekday"].shift(1)
        out["prior_pressure"] = out["pressure"].shift(1)
        valid = np.isfinite(out["prior_pressure"]) & np.isfinite(out["intraday_return"])
        out["transmission"] = np.nan
        for weekday in range(5):
            selected = out["prior_weekday"].eq(weekday)
            out[f"cross_{weekday}"] = (out["prior_pressure"] * out["intraday_return"]).where(selected, 0).where(valid)
            out[f"energy_{weekday}"] = out["prior_pressure"].pow(2).where(selected, 0).where(valid)
            out[f"slope_{weekday}"] = out[f"cross_{weekday}"].rolling(params.season_window).sum() / out[f"energy_{weekday}"].rolling(params.season_window).sum().where(lambda s: s > 1e-12)
            today = out["weekday"].eq(weekday)
            out.loc[today, "transmission"] = out.loc[today, f"slope_{weekday}"]
        out["expected_return"] = out["pressure"] * out["transmission"]
        out["strength"] = (out["expected_return"] / out["atr_fraction"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeSeasonParams) -> SignalFrame:
        strength = indicators["strength"].fillna(0).to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        direction = 0
        entry_price = np.nan
        stop_distance = np.nan
        held = 0
        exposure = 0.25
        for i in range(len(data)):
            if direction:
                # A prior-close decision fills at this session's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    stop_distance = params.stop_atr * atr[i - 1]
                held += 1
                invalid = not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(closes[i])
                stopped = direction * (closes[i] - entry_price) < -stop_distance
                if invalid or stopped or held >= params.holding_bars:
                    direction = 0
                    entry_price = np.nan
                    held = 0
                    # Remain flat for the following session after an exit.
                    continue
                signals[i] = direction
                sizes[i] = exposure
            elif np.isfinite(strength[i]) and abs(strength[i]) >= params.min_strength and np.isfinite(atr[i]) and atr[i] > 0:
                direction = 1 if strength[i] > 0 else -1
                entry_price = np.nan
                held = 0
                exposure = float(np.clip(abs(strength[i]) / (4 * params.min_strength), 0.25, 0.95))
                signals[i] = direction
                sizes[i] = exposure
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RangeSeasonParams(stop_atr=1.0)
    index = pd.date_range("2024-01-01", periods=4)
    data = pd.DataFrame({"open": [90.0, 100.0, 100.0, 100.0], "close": [90.0, 100.0, 100.0, 100.0]}, index=index)
    ind = pd.DataFrame({"strength": [0.16, 0.0, 0.0, 0.0], "atr": [2.0] * 4}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    assert np.isclose(result["size"].iloc[1], 0.5)
    data.loc[index[1], "close"] = 97.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0]
    ind.loc[index[0], "strength"] = -0.16
    data.loc[index[1], "close"] = 103.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, -1, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
