from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoupledSpringParams:
    ma_window: int = 20
    coupling_window: int = 40
    gap_window: int = 3
    z_entry: float = 1.1
    coupling_floor: float = 0.2
    gap_floor: float = 0.002
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[CoupledSpringParams]):
    strategy_id = "gen_local_1791213614"

    @classmethod
    def params_type(cls) -> type[CoupledSpringParams]:
        return CoupledSpringParams

    def warmup_bars(self, params: CoupledSpringParams) -> int:
        return int(max(params.ma_window, params.coupling_window + 1, params.gap_window + 1))

    def indicators(self, data: pd.DataFrame, params: CoupledSpringParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["dispersion"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["dispersion"].replace(0.0, np.nan)
        out["overnight_return"] = opening / close.shift(1) - 1.0
        out["intraday_return"] = close / opening - 1.0
        out["coupling"] = out["overnight_return"].rolling(
            params.coupling_window, min_periods=params.coupling_window
        ).corr(out["intraday_return"])
        out["overnight_pressure"] = out["overnight_return"].rolling(
            params.gap_window, min_periods=params.gap_window
        ).sum()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: CoupledSpringParams,
    ) -> SignalFrame:
        z = indicators["z"]
        coupling = indicators["coupling"]
        pressure = indicators["overnight_pressure"]
        valid = np.isfinite(z) & np.isfinite(coupling) & np.isfinite(pressure)
        following = coupling >= params.coupling_floor
        fading = coupling <= -params.coupling_floor
        long_displacement = (following & (z >= params.z_entry)) | (fading & (z <= -params.z_entry))
        short_displacement = (following & (z <= -params.z_entry)) | (fading & (z >= params.z_entry))
        entries = pd.Series(0, index=data.index, dtype=int)
        entries.loc[valid & long_displacement & (pressure >= params.gap_floor)] = 1
        entries.loc[valid & short_displacement & (pressure <= -params.gap_floor)] = -1

        raw = np.zeros(len(data), dtype=int)
        opening = data["open"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        candidates = entries.to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # The preceding close requested entry; its fill is today's open.
                if i == entry_bar:
                    entry_price = opening[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        continue
                held_bars = i - entry_bar + 1
                gain = position * (close[i] / entry_price - 1.0)
                # Targets are observed at the close and exited at the next open.
                if gain >= params.profit_target or held_bars >= params.max_hold:
                    position = 0
                    continue
            elif candidates[i] and np.isfinite(close[i]) and close[i] > 0:
                position = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
            raw[i] = position

        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = CoupledSpringParams(profit_target=0.02, max_hold=3)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": 100.0, "high": 115.0, "low": 95.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"z": 2.0, "coupling": 0.5, "overnight_pressure": 0.01}, index=index)
    run = lambda bars, features: GeneratedStrategy.generate_signals(None, bars, features, None, params).data
    assert run(data, ind)["signal"].tolist() == [0, 1, 1, 1, 0, 1]
    data.loc[index[1], ["open", "close"]] = 110.0
    data.loc[index[2], "close"] = 113.0
    assert run(data, ind)["signal"].tolist() == [0, 1, 1, 0, 1, 1]
    data.loc[index[2], "close"] = 107.0
    ind[["z", "overnight_pressure"]] *= -1
    assert run(data, ind)["signal"].tolist() == [0, -1, -1, 0, -1, -1]
    assert (run(data, ind * np.nan)["signal"] == 0).all()
    assert (run(data, ind)["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
