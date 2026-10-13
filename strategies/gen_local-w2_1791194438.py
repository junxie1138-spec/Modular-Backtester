from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownShelfParams:
    peak_window: int = 63
    shelf_window: int = 6
    atr_window: int = 14
    min_depth_atr: float = 3.0
    shelf_width_atr: float = 1.2
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[DrawdownShelfParams]):
    strategy_id = "gen_local-w2_1791194438"

    @classmethod
    def params_type(cls) -> type[DrawdownShelfParams]:
        return DrawdownShelfParams

    def warmup_bars(self, params: DrawdownShelfParams) -> int:
        return int(max(params.peak_window + params.shelf_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownShelfParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["peak"] = close.rolling(params.peak_window).max()
        out["depth"] = 1.0 - close / out["peak"].replace(0.0, np.nan)
        out["atr_fraction"] = out["atr"] / out["peak"].replace(0.0, np.nan)
        prior_depth = out["depth"].shift(1).rolling(params.shelf_window)
        out["shelf_floor"] = prior_depth.min()
        out["shelf_ceiling"] = prior_depth.max()
        out["shelf_width"] = out["shelf_ceiling"] - out["shelf_floor"]
        prior_peak = out["peak"].shift(1).rolling(params.shelf_window)
        out["prior_peak_min"] = prior_peak.min()
        out["prior_peak_max"] = prior_peak.max()
        out["stable_peak"] = (
            out["peak"].eq(out["prior_peak_min"])
            & out["prior_peak_min"].eq(out["prior_peak_max"])
        )
        scale = out["atr_fraction"].where(out["atr_fraction"] > 0.0)
        out["conviction"] = (
            out["shelf_floor"] / (2.0 * params.min_depth_atr * scale)
            * (1.0 - 0.5 * out["shelf_width"] / (params.shelf_width_atr * scale))
        ).replace([np.inf, -np.inf], np.nan).clip(0.25, 0.95).fillna(0.25)
        out["entry"] = (
            out["stable_peak"]
            & (scale > 0.0)
            & (out["shelf_floor"] >= params.min_depth_atr * scale)
            & (out["shelf_width"] <= params.shelf_width_atr * scale)
            & (out["depth"] < out["shelf_floor"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: DrawdownShelfParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.25}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].to_numpy(dtype=float)
        pending = active = False
        entry_atr = stop = 0.0
        trade_size = 0.25
        held = 0
        for i in range(len(data)):
            if pending:
                pending = False
                active = bool(np.isfinite(opening[i]) and opening[i] > 0.0)
                # Freeze the decision-bar ATR against the next-bar entry open.
                stop = opening[i] - params.stop_atr * entry_atr
                held = 0
            if active:
                held += 1
                if not np.isfinite(close[i]) or close[i] < stop or held >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = trade_size
                continue
            if (entries[i] and np.isfinite(close[i]) and close[i] > 0.0
                    and np.isfinite(atr[i]) and atr[i] > 0.0
                    and np.isfinite(conviction[i])):
                entry_atr = atr[i]
                trade_size = float(np.clip(conviction[i], 0.25, 0.95))
                signals[i] = 1
                sizes[i] = trade_size
                pending = True
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = DrawdownShelfParams()
    index = pd.date_range("2020-01-01", periods=25)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    ind = pd.DataFrame({"atr": 2.0, "entry": False, "conviction": 0.4}, index=index)
    ind.loc[index[[0, 22]], "entry"] = True
    ind.loc[index[22], "conviction"] = 0.9
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[0] == 0
    assert (result["signal"].iloc[1:19] == 1).all()
    assert (result["signal"].iloc[19:23] == 0).all()
    assert result["size"].iloc[1] == 0.4 and result["size"].iloc[23] == 0.9
    data.loc[index[2], "close"] = 96.0
    data.loc[index[3], "close"] = 94.0
    ind.loc[index[2]:, "atr"] = 100.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[:5].tolist() == [0, 1, 1, 1, 0]
    assert (result["size"] > 0.0).all()
    assert set(result["signal"].unique()) <= {0, 1}
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
