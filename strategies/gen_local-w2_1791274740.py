from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCouplingParams:
    regime_window: int = 40
    atr_window: int = 14
    coupling_threshold: float = 0.15
    min_gap_atr: float = 0.20
    max_gap_atr: float = 1.50
    stop_atr: float = 1.80
    holding_days: int = 4


class GeneratedStrategy(BaseStrategy[GapCouplingParams]):
    strategy_id = "gen_local-w2_1791274740"

    @classmethod
    def params_type(cls) -> type[GapCouplingParams]:
        return GapCouplingParams

    def warmup_bars(self, params: GapCouplingParams) -> int:
        return max(params.regime_window + 2, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapCouplingParams) -> pd.DataFrame:
        if not (params.regime_window >= 2 and params.atr_window >= 1
                and 3 <= params.holding_days <= 5
                and 0 < params.coupling_threshold
                and 0 < params.min_gap_atr < params.max_gap_atr
                and params.stop_atr > 0):
            raise ValueError("Invalid gap-coupling parameters")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(data["open"] > 0)
        out["gap"] = opening / out["prior_close"] - 1.0
        out["day_return"] = data["close"] / opening - 1.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["prior_atr"] = out["atr"].shift(1).where(lambda s: s > 0)
        out["gap_atr"] = (opening - out["prior_close"]) / out["prior_atr"]
        w = params.regime_window
        out["mean_gap"] = out["gap"].rolling(w).mean()
        out["mean_day"] = out["day_return"].rolling(w).mean()
        out["mean_product"] = (out["gap"] * out["day_return"]).rolling(w).mean()
        out["mean_gap_squared"] = out["gap"].pow(2).rolling(w).mean()
        out["gap_variance"] = (out["mean_gap_squared"] - out["mean_gap"].pow(2)).clip(lower=0)
        out["covariance"] = out["mean_product"] - out["mean_gap"] * out["mean_day"]
        out["coupling"] = (out["covariance"] / out["gap_variance"].where(
            out["gap_variance"] > 1e-12)).shift(1).clip(-3.0, 3.0)
        required = ["coupling", "gap_atr", "day_return", "atr"]
        out["ready"] = np.isfinite(out[required]).all(axis=1) & (out["atr"] > 0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapCouplingParams) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        regime = 0
        pending = False
        active = False
        entry_atr = 0.0
        stop = 0.0
        held = 0
        trade_size = 1.0
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        beta = indicators["coupling"].to_numpy(dtype=float)
        gaps = indicators["gap_atr"].to_numpy(dtype=float)
        days = indicators["day_return"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        for i in range(len(data)):
            if np.isfinite(beta[i]):
                if beta[i] > params.coupling_threshold:
                    regime = 1
                elif beta[i] < -params.coupling_threshold:
                    regime = -1
            if pending:
                pending = False
                active = True
                held = 0
                # The prior close decision fills at this bar's open.
                stop = opening[i] - params.stop_atr * entry_atr
            if active:
                held += 1
                if (not np.isfinite(opening[i]) or opening[i] <= 0
                        or not np.isfinite(closing[i])
                        or closing[i] < stop or held >= params.holding_days):
                    active = False
                else:
                    signals[i] = 1
                sizes[i] = trade_size
                continue
            if (ready[i] and regime != 0 and regime * gaps[i] > 0
                    and params.min_gap_atr <= abs(gaps[i]) <= params.max_gap_atr
                    and days[i] > 0):
                trade_size = float(np.clip(abs(beta[i] * gaps[i]), 0.25, 1.0))
                entry_atr = atr[i]
                pending = True
                signals[i] = 1
                sizes[i] = trade_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "coupling": 0.8, "gap_atr": 0.5, "day_return": 0.01,
        "atr": 100.0, "ready": False,
    }, index=index)
    ind.loc[index[0], ["atr", "ready"]] = [1.0, True]
    strategy = GeneratedStrategy()
    params = GapCouplingParams()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.4)
    data.loc[index[2], "close"] = 98.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
