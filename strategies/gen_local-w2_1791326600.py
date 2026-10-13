from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MirrorEnergyParams:
    hold_bars: int = 18
    energy_distance: float = 0.20
    damping_ratio: float = 0.80


class GeneratedStrategy(BaseStrategy[MirrorEnergyParams]):
    strategy_id = "gen_local-w2_1791326600"

    @classmethod
    def params_type(cls):
        return MirrorEnergyParams

    def warmup_bars(self, params: MirrorEnergyParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: MirrorEnergyParams) -> pd.DataFrame:
        if not 15 <= params.hold_bars <= 20:
            raise ValueError("hold_bars must be between 15 and 20")
        if not 0 < params.energy_distance <= 2 or not 0 < params.damping_ratio <= 1:
            raise ValueError("Invalid energy_distance or damping_ratio")
        out = pd.DataFrame(index=data.index)
        ratio = data["close"].where(data["close"] > 0) / data["open"].where(data["open"] > 0)
        out["intraday_return"] = np.log(ratio).replace([np.inf, -np.inf], np.nan)
        r = out["intraday_return"]
        # Orthogonal four-session Haar transform: two fine details, one coarse detail, one level.
        out["fine_early"] = (r.shift(3) - r.shift(2)) / np.sqrt(2.0)
        out["fine_late"] = (r.shift(1) - r) / np.sqrt(2.0)
        out["coarse"] = (r.shift(3) + r.shift(2) - r.shift(1) - r) / 2.0
        out["level"] = (r.shift(3) + r.shift(2) + r.shift(1) + r) / 2.0
        components = ["fine_early", "fine_late", "coarse", "level"]
        energy_columns = []
        for name in components:
            col = name + "_energy"
            out[col] = out[name].pow(2)
            energy_columns.append(col)
        out["total_energy"] = out[energy_columns].sum(axis=1, min_count=4)
        denominator = out["total_energy"].where(out["total_energy"] > 1e-12)
        distances = []
        for name in components:
            share = name + "_share"
            prior = "prior_" + share
            distance = name + "_distance"
            out[share] = out[name + "_energy"] / denominator
            out[prior] = out[share].shift(4)
            out[distance] = (out[share] - out[prior]).abs()
            distances.append(distance)
        out["energy_distance"] = out[distances].sum(axis=1, min_count=4)
        out["prior_energy"] = out["total_energy"].shift(4)
        out["energy_ratio"] = out["total_energy"] / out["prior_energy"].where(out["prior_energy"] > 1e-12)
        out["prior_level"] = out["level"].shift(4)
        out["entry"] = (
            (out["energy_distance"] <= params.energy_distance)
            & (out["energy_ratio"] <= params.damping_ratio)
            & (out["prior_level"] > 0)
            & (out["level"] < 0)
            & (r > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MirrorEnergyParams,
    ) -> SignalFrame:
        if not 15 <= params.hold_bars <= 20:
            raise ValueError("hold_bars must be between 15 and 20")
        eligible = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    # Always publish a flat bar before permitting another entry.
                    exit_at = -1
            elif eligible[i]:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = MirrorEnergyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=64, freq="B")
    returns = np.tile([0.012, 0.008, 0.003, -0.003, -0.006, -0.004, -0.0015, 0.0015], 8)
    close = 100.0 * np.exp(returns)
    data = pd.DataFrame(
        {"open": 100.0, "high": np.maximum(100.0, close), "low": np.minimum(100.0, close),
         "close": close, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
    features = strategy.indicators(data, params)
    assert not features["entry"].iloc[:7].any()
    assert features["entry"].iloc[7]
    pd.testing.assert_frame_equal(features.iloc[:32], strategy.indicators(data.iloc[:32], params))
    forced = pd.DataFrame({"entry": True}, index=index)
    signals = strategy.generate_signals(data, forced, ctx, params).data
    assert signals["signal"].iloc[0] == 0
    assert signals["signal"].iloc[1:19].eq(1).all()
    assert signals["signal"].iloc[19] == 0
    assert signals["signal"].iloc[20] == 1
    assert signals["size"].gt(0).all()
    flat = data.copy()
    flat["close"] = 100.0
    assert not strategy.indicators(flat, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
