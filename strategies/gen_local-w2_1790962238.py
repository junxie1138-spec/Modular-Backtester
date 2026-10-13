from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    frac_order: float = 0.45
    frac_window: int = 32
    memory_window: int = 40
    corr_floor: float = 0.10
    echo_floor: float = 0.15
    body_floor: float = 0.20
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local-w2_1790962238"

    @classmethod
    def params_type(cls):
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(params.frac_window + params.memory_window + 2)

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if not (0.0 < params.frac_order < 1.0 and params.frac_window >= 2
                and params.memory_window >= 3 and params.hold_bars in (1, 2)
                and 0.0 <= params.corr_floor < 1.0 and params.echo_floor >= 0.0
                and 0.0 <= params.body_floor < 1.0):
            raise ValueError("Invalid fractional echo parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        weights = np.ones(params.frac_window)
        for k in range(1, params.frac_window):
            weights[k] = weights[k - 1] * (k - 1 - params.frac_order) / k
        # ponytail: finite fractional filter; increase frac_window for longer memory.
        out["frac_close"] = out["log_close"] * weights[0]
        for k in range(1, params.frac_window):
            out["frac_close"] += weights[k] * out["log_close"].shift(k)
        out["impulse"] = out["frac_close"].diff()
        out["lag1"] = out["impulse"].shift(1)
        out["lag2"] = out["impulse"].shift(2)
        out["scale"] = out["impulse"].rolling(params.memory_window).std(ddof=0)
        out["rho1"] = out["impulse"].rolling(params.memory_window).corr(out["lag1"])
        out["rho2"] = out["impulse"].rolling(params.memory_window).corr(out["lag2"])
        out["echo"] = out["rho2"] * out["lag1"] / out["scale"].where(out["scale"] > 0.0)
        out["range"] = data["high"] - data["low"]
        out["body"] = (data["close"] - data["open"]) / out["range"].where(out["range"] > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        delayed = (indicators["rho2"].abs() > params.corr_floor) & (
            indicators["rho2"].abs() > indicators["rho1"].abs())
        longs = delayed & (indicators["echo"] > params.echo_floor) & (
            indicators["body"] > params.body_floor)
        shorts = delayed & (indicators["echo"] < -params.echo_floor) & (
            indicators["body"] < -params.body_floor)
        entries = np.select([longs.fillna(False), shorts.fillna(False)], [1, -1], default=0)
        target = np.zeros(len(data), dtype=int)
        side, exit_bar = 0, -1
        for i, entry in enumerate(entries):
            if side != 0:
                if i == exit_bar:
                    side = 0
                # Ignore new entries while holding and on the exit bar.
            elif entry != 0:
                side = int(entry)
                exit_bar = i + params.hold_bars
            target[i] = side
        df = pd.DataFrame({"signal": target, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    dates = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=dates)
    votes = [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    ind = pd.DataFrame({"rho1": 0.0, "rho2": 0.5, "echo": votes, "body": votes}, index=dates)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    assert strategy.generate_signals(data, ind, ctx, EchoParams()).data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    ind["body"] = -ind["body"]
    assert strategy.generate_signals(data, ind, ctx, EchoParams()).data["signal"].eq(0).all()
