from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    memory: int = 63
    atr_period: int = 14
    min_correlation: float = 0.15
    entry_score: float = 0.60
    hold_bars: int = 18
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = "gen_local-w2_1791235685"

    @classmethod
    def params_type(cls):
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return int(max(params.memory + 3, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        if (
            params.memory < 3 or params.atr_period < 1 or params.hold_bars < 1
            or not 0.0 <= params.min_correlation < 0.85
            or not np.isfinite(params.entry_score) or params.entry_score <= 0.0
            or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = (
            data["open"] / out["previous_close"].replace(0.0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out["gap_scale"] = out["gap"].shift(1).rolling(
            params.memory, min_periods=params.memory
        ).std(ddof=0)
        out["correlation"] = out["gap"].shift(1).rolling(
            params.memory, min_periods=params.memory
        ).corr(out["gap"].shift(2)).replace([np.inf, -np.inf], np.nan)
        out["rho"] = out["correlation"].clip(-0.85, 0.85)
        out["gap_z"] = out["gap"] / out["gap_scale"].where(out["gap_scale"] > 0.0)
        out["projection"] = (
            out["rho"] * (1.0 - out["rho"] ** params.hold_bars)
            / (1.0 - out["rho"])
        )
        out["score"] = (out["gap_z"] * out["projection"]).replace(
            [np.inf, -np.inf], np.nan
        )
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - out["previous_close"]).abs(),
                (data["low"] - out["previous_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        valid = (
            out["score"].notna()
            & out["correlation"].abs().ge(params.min_correlation)
            & out["score"].abs().ge(params.entry_score)
            & out["atr"].gt(0.0)
            & np.isfinite(out["atr"])
        )
        out["candidate"] = np.sign(out["score"].fillna(0.0)).where(valid, 0).astype(int)
        out["conviction_size"] = (
            0.20 + 0.75 * (out["score"].abs().fillna(0.0) / 3.0).clip(0.0, 1.0)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapMemoryParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.20, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        atrs = indicators["atr"].to_numpy(dtype=float)
        convictions = indicators["conviction_size"].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        position_size = 0.20
        for i in range(count):
            if side:
                if i == entry_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        side = 0
                    else:
                        stop = opens[i] - side * params.stop_atr * entry_atr
                if side and (
                    not np.isfinite(closes[i])
                    or side * (closes[i] - stop) < 0.0
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    side = 0
            elif (
                candidates[i] != 0
                and np.isfinite(atrs[i]) and atrs[i] > 0.0
                and np.isfinite(convictions[i]) and convictions[i] > 0.0
                and np.isfinite(closes[i]) and closes[i] > 0.0
            ):
                side = int(candidates[i])
                entry_bar = i + 1
                entry_atr = float(atrs[i])
                position_size = float(np.clip(convictions[i], 0.20, 0.95))
            signals[i] = side
            sizes[i] = position_size if side else 0.20
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": [100.0, 101.0, 100.0, 100.0, 99.0, 100.0, 100.0, 100.0],
         "close": [100.0, 100.0, 97.0, 100.0, 100.0, 103.0, 100.0, 100.0]},
        index=index,
    )
    ind = pd.DataFrame(
        {"candidate": [1, 0, 0, -1, 0, 0, 0, 0],
         "atr": 1.0, "conviction_size": 0.70},
        index=index,
    )
    params = GapMemoryParams(hold_bars=3, stop_atr=2.0)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 0]
    assert result["size"].gt(0.0).all() and result["size"].iloc[1] == 0.70
    data.loc[:, ["open", "close"]] = 100.0
    ind["candidate"] = [1, 0, 0, 0, 0, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
