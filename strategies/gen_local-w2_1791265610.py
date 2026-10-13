from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    history: int = 252
    edge: float = 0.0005


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791265610"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.history) + 8

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if params.history < 32 or not np.isfinite(params.edge) or params.edge < 0:
            raise ValueError("history must be >= 32 and edge must be finite and >= 0")
        out = pd.DataFrame(index=data.index)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        columns = []
        for lag in (3, 2, 1, 0):
            name = f"return_lag_{lag}"
            out[name] = returns.shift(lag)
            columns.append(name)
        raw = out[columns].to_numpy(dtype=float)
        scale = np.sqrt(np.mean(raw * raw, axis=1))
        valid = np.isfinite(raw).all(axis=1) & np.isfinite(scale) & (scale > 1e-12)
        shapes = np.zeros_like(raw)
        np.divide(raw, scale[:, None], out=shapes, where=valid[:, None])
        out["shape_scale"] = np.where(valid, scale, 0.0)
        for j in range(4):
            out[f"shape_{j}"] = shapes[:, j]
        r = returns.to_numpy(dtype=float)
        direct = np.zeros(len(data), dtype=float)
        mirror = np.zeros(len(data), dtype=float)
        ready = np.zeros(len(data), dtype=bool)
        # ponytail: O(bars * history); batch distances if much longer archives are needed.
        for t in range(self.warmup_bars(params), len(data)):
            if not valid[t]:
                continue
            # Historical shapes and their next-day outcomes precede the current shape.
            lo, hi = t - int(params.history) - 4, t - 4
            labels = r[lo + 1:hi + 1]
            usable = valid[lo:hi] & np.isfinite(labels)
            if usable.sum() < 32:
                continue
            archive = shapes[lo:hi][usable]
            outcomes = labels[usable]
            distance = np.sum((archive - shapes[t]) ** 2, axis=1)
            mirror_distance = np.sum((archive + shapes[t]) ** 2, axis=1)
            near = np.argpartition(distance, 7)[:8]
            opposite = np.argpartition(mirror_distance, 7)[:8]
            direct[t] = outcomes[near].mean()
            mirror[t] = outcomes[opposite].mean()
            ready[t] = True
        out["direct_forecast"] = direct
        out["mirror_forecast"] = mirror
        out["edge_forecast"] = 0.5 * (direct - mirror)
        out["ready"] = ready
        out[columns] = out[columns].fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        entry = indicators["ready"].fillna(False) & (
            indicators["edge_forecast"].fillna(0.0) > params.edge
        )
        df = pd.DataFrame(index=data.index)
        # Re-evaluate the one-day forecast daily; exit only when entry flips false.
        df["signal"] = entry.astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AnalogParams(history=32)
    n = 160
    rng = np.random.default_rng(19)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    data = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:120], params)
    pd.testing.assert_frame_equal(ind.iloc[:120], prefix)
    forced = ind.copy()
    forced["ready"] = True
    forced["edge_forecast"] = np.resize(np.array([0.001, 0.001, -0.001]), n)
    result = strategy.generate_signals(data, forced, ctx, params).data
    expected = (forced["edge_forecast"] > params.edge).astype(int).shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(result["signal"], expected, check_names=False)
    assert result["signal"].isin([0, 1]).all() and (result["size"] > 0).all()
    flat = data.copy()
    flat["close"] = 100.0
    flat_ind = strategy.indicators(flat, params)
    assert np.isfinite(flat_ind.select_dtypes(include="number").to_numpy()).all()
    assert not strategy.generate_signals(flat, flat_ind, ctx, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
