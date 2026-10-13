from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideParams:
    baseline_bars: int = 20
    regime_bars: int = 10
    efficiency_threshold: float = 0.35
    volume_threshold: float = 1.25
    move_threshold: float = 0.8
    profit_target: float = 0.012
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[TideParams]):
    strategy_id = "gen_local_1791257014"

    @classmethod
    def params_type(cls):
        return TideParams

    def warmup_bars(self, params: TideParams) -> int:
        return int(max(params.baseline_bars, params.regime_bars) + 2)

    def indicators(self, data: pd.DataFrame, params: TideParams) -> pd.DataFrame:
        if (
            min(params.baseline_bars, params.regime_bars) < 2
            or not 0 < params.efficiency_threshold < 1
            or params.volume_threshold <= 1
            or params.move_threshold <= 0
            or params.profit_target <= 0
            or params.max_hold_bars not in (1, 2)
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        past = out["return"].shift(1)
        out["noise"] = past.rolling(params.baseline_bars).std(ddof=0)
        volume_base = data["volume"].shift(1).rolling(params.baseline_bars).mean()
        out["volume_base"] = volume_base
        out["volume_ratio"] = data["volume"] / volume_base.where(volume_base > 0)
        out["net_return"] = past.rolling(params.regime_bars).sum()
        out["path_length"] = past.abs().rolling(params.regime_bars).sum()
        out["efficiency"] = out["net_return"].abs() / out["path_length"].where(out["path_length"] > 0)
        out["range"] = data["high"] - data["low"]
        out["close_location"] = (data["close"] - data["low"]) / out["range"].where(out["range"] > 0)
        out["move_strength"] = out["return"].abs() / out["noise"].where(out["noise"] > 0)
        out["direction"] = np.sign(out["return"])
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        ready = out[["volume_ratio", "efficiency", "close_location", "move_strength", "direction"]].notna().all(axis=1)
        confirmed = ready & (out["volume_ratio"] >= params.volume_threshold) & (out["move_strength"] >= params.move_threshold)
        trending = out["efficiency"] >= params.efficiency_threshold
        aligned = out["direction"] * out["net_return"] > 0
        accepted = out["direction"] * (out["close_location"] - 0.5) >= 0.2
        rejected = out["direction"] * (out["close_location"] - 0.5) <= -0.1
        out["candidate"] = 0
        follow = confirmed & trending & aligned & accepted
        fade = confirmed & ~trending & rejected
        out.loc[follow, "candidate"] = out.loc[follow, "direction"].astype(int)
        out.loc[fade, "candidate"] = -out.loc[fade, "direction"].astype(int)
        out["conviction"] = (
            out["move_strength"] / (2 * params.move_threshold)
            * out["volume_ratio"] / params.volume_threshold
        ).clip(0.2, 1.0).fillna(0.2)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.2, dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        conviction = indicators["conviction"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction = 0
        age = 0
        entry = 0.0
        held_size = 0.2
        for i in range(len(data)):
            if direction:
                # The previous close's decision fills at this bar's open.
                if age == 0:
                    entry = opens[i]
                age += 1
                gain = direction * (closes[i] / entry - 1.0) if entry > 0 else 0.0
                if gain >= params.profit_target or age >= params.max_hold_bars:
                    direction = 0
                    age = 0
                    continue
            elif candidates[i]:
                direction = int(candidates[i])
                held_size = float(conviction[i])
                age = 0
            signal[i] = direction
            size[i] = held_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = TideParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        features = pd.DataFrame(
            {"candidate": [direction, 0, 0, 0, 0], "conviction": 0.7}, index=index
        )
        timed = strategy.generate_signals(data, features, ctx, params).data
        assert timed["signal"].tolist() == [0, direction, direction, 0, 0]
        assert timed["size"].iloc[1] == 0.7
        profitable = data.copy()
        profitable.loc[index[1], "close"] = 100.0 * (1 + direction * 0.02)
        targeted = strategy.generate_signals(profitable, features, ctx, params).data
        assert targeted["signal"].tolist() == [0, direction, 0, 0, 0]
    warm = strategy.indicators(data, params)
    assert warm["candidate"].eq(0).all()
    assert warm["conviction"].gt(0).all()


if __name__ == "__main__":
    _self_check()
