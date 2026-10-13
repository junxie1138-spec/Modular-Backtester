from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    energy_window: int = 4
    position_window: int = 16
    energy_floor: float = 0.20
    location_tail: float = 0.35


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791282617"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.position_window, 8 + params.energy_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.energy_window < 1 or params.position_window < 8:
            raise ValueError("Windows must be positive; position_window must be at least 8.")
        if not 0.0 < params.energy_floor < 1.0 or not 0.0 < params.location_tail < 0.5:
            raise ValueError("Invalid energy_floor or location_tail.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (1, 2, 4):
            width = 2 * half
            recent = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{width}"] = np.sqrt(half / 2.0) * (recent - recent.shift(half))
            out[f"energy_{width}"] = out[f"haar_{width}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out["energy_2"] + out["energy_4"] + out["energy_8"]
        out["fine_share"] = out["energy_2"] / out["total_energy"].where(out["total_energy"] > 0)
        out["previous_share"] = out["fine_share"].shift(2)
        out["channel_low"] = data["low"].rolling(params.position_window, min_periods=params.position_window).min()
        out["channel_high"] = data["high"].rolling(params.position_window, min_periods=params.position_window).max()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["relative_position"] = (
            (data["close"] - out["channel_low"]) / out["channel_width"].where(out["channel_width"] > 0)
        ).clip(0.0, 1.0)
        out["ready"] = np.isfinite(out).all(axis=1)
        energy_turn = (out["fine_share"] >= params.energy_floor) & (out["fine_share"] > out["previous_share"])
        out["wavelet_long"] = energy_turn & (out["haar_2"] > 0) & (out["haar_8"] < 0)
        out["wavelet_short"] = energy_turn & (out["haar_2"] < 0) & (out["haar_8"] > 0)
        out["location_long"] = out["relative_position"] <= params.location_tail
        out["location_short"] = out["relative_position"] >= 1.0 - params.location_tail
        out["long_entry"] = out["ready"] & out["wavelet_long"] & out["location_long"]
        out["short_entry"] = out["ready"] & out["wavelet_short"] & out["location_short"]
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        candidates = np.select(
            [indicators["long_entry"].astype(bool), indicators["short_entry"].astype(bool)],
            [1, -1], default=0
        )
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    evidence = pd.DataFrame({
        "long_entry": [True, False, False, False, False, True, False],
        "short_entry": [False, False, True, True, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, evidence, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    index = pd.date_range("2020-01-01", periods=80)
    close = 100.0 + 3.0 * np.sin(np.arange(80) * np.pi / 8.0)
    data = pd.DataFrame({
        "open": close, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": 1000.0,
    }, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:50], params)
    pd.testing.assert_frame_equal(full.iloc[:50], prefix)
    assert np.isfinite(full.to_numpy(dtype=float)).all()
    assert not full["long_entry"].iloc[:strategy.warmup_bars(params) - 1].any()
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    flat_indicators = strategy.indicators(flat, params)
    assert not flat_indicators[["long_entry", "short_entry"]].to_numpy().any()


if __name__ == "__main__":
    _self_check()
