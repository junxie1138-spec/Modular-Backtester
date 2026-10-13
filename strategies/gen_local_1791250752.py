from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 20
    occupancy_window: int = 15
    calibration_window: int = 126
    tail_quantile: float = 0.20
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791250752"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.ma_window + params.occupancy_window + params.calibration_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.ma_window, params.occupancy_window, params.calibration_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.0 < params.tail_quantile < 0.5:
            raise ValueError("tail_quantile must be between zero and 0.5.")
        if params.profit_target <= 0.0 or params.max_hold_bars not in (1, 2):
            raise ValueError("Use a positive target and a one- or two-bar time stop.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["price_std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        history = out["z"].shift(1).rolling(params.calibration_window)
        out["z_low"] = history.quantile(params.tail_quantile)
        out["z_high"] = history.quantile(1.0 - params.tail_quantile)
        out["occupancy"] = (out["z"] > 0.0).astype(float).where(out["z"].notna()).rolling(params.occupancy_window).mean()
        capacity = out["occupancy"].shift(1).rolling(params.calibration_window)
        out["occupancy_low"] = capacity.quantile(0.25)
        out["occupancy_high"] = capacity.quantile(0.75)
        ready = out.notna().all(axis=1)
        saturated = out["occupancy"] > out["occupancy_high"]
        balanced = (out["occupancy"] >= out["occupancy_low"]) & (out["occupancy"] <= out["occupancy_high"])
        breakout = (out["z"] > out["z_high"]) & (out["z"].shift(1) <= out["z_high"].shift(1)) & (out["z"] > 0.0)
        recoil = (out["z"].shift(1) <= out["z_low"].shift(1)) & (out["z"] > out["z"].shift(1)) & (out["z"] < 0.0)
        out["entry"] = (ready & ((saturated & breakout) | (balanced & recoil))).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Learn the fill price only on the scheduled entry bar.
                if i == entry_bar:
                    entry_price = opens[i]
                target_hit = np.isfinite(entry_price) and entry_price > 0.0 and closes[i] >= entry_price * (1.0 + params.profit_target)
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    raw[i] = 1
                # Keep the exit decision flat even if another entry qualifies.
                continue
            if entries[i]:
                raw[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({name: 100.0 for name in ("open", "high", "low", "close", "volume")}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = Params()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    signals = strategy.generate_signals(data, indicators, ctx, params).data["signal"]
    assert signals.tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 102.0
    signals = strategy.generate_signals(data, indicators, ctx, params).data["signal"]
    assert signals.tolist() == [0, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
