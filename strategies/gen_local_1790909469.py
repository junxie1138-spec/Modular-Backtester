from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeCycleParams:
    shape_bars: int = 6
    history_bars: int = 252
    hold_bars: int = 8
    analog_percentile: float = 0.15
    min_analogs: int = 12


class GeneratedStrategy(BaseStrategy[RangeCycleParams]):
    strategy_id = "gen_local_1790909469"

    @classmethod
    def params_type(cls):
        return RangeCycleParams

    def warmup_bars(self, params: RangeCycleParams) -> int:
        return params.history_bars + params.shape_bars + params.hold_bars + 3

    def indicators(self, data: pd.DataFrame, params: RangeCycleParams) -> pd.DataFrame:
        if (params.shape_bars < 3 or params.history_bars < params.min_analogs
                or params.hold_bars < 1 or params.min_analogs < 3
                or not 0.0 < params.analog_percentile <= 1.0):
            raise ValueError("Invalid range-cycle parameters")
        out = pd.DataFrame(index=data.index)
        out["range_fraction"] = (
            (data["high"] - data["low"]).where(data["high"] > data["low"])
            / data["close"].where(data["close"] > 0.0)
        )
        out["log_range"] = np.log(out["range_fraction"])
        out["curvature"] = out["log_range"].diff().diff()
        out["energy"] = np.sqrt(
            out["curvature"].pow(2).rolling(params.shape_bars).sum()
        ).replace(0.0, np.nan)
        columns = []
        for lag in range(params.shape_bars):
            name = f"shape_{lag}"
            columns.append(name)
            out[name] = out["curvature"].shift(lag) / out["energy"]
        # At bar k this is the observed return from open[k-h] to open[k].
        out["matured_return"] = (
            data["open"].where(data["open"] > 0.0)
            / data["open"].shift(params.hold_bars).where(
                data["open"].shift(params.hold_bars) > 0.0
            ) - 1.0
        )
        shapes = out[columns].to_numpy(dtype=float)
        matured = out["matured_return"].to_numpy(dtype=float)
        valid = np.isfinite(shapes).all(axis=1)
        lower = np.full(len(data), np.nan)
        upper = np.full(len(data), np.nan)
        cutoff = np.full(len(data), np.nan)
        counts = np.zeros(len(data), dtype=int)
        # ponytail: O(bars * history * shape); batch distances if histories become large.
        for t in range(self.warmup_bars(params) - 1, len(data)):
            if not valid[t]:
                continue
            end = t - max(params.shape_bars + 2, params.hold_bars + 1)
            candidates = np.arange(
                max(params.shape_bars + 1, end - params.history_bars + 1),
                end + 1,
            )
            # Endpoint j's entry is j+1 and exit is j+1+h; both are observed by t.
            outcomes = matured[candidates + params.hold_bars + 1]
            eligible = valid[candidates] & np.isfinite(outcomes)
            candidates = candidates[eligible]
            outcomes = outcomes[eligible]
            if len(candidates) < params.min_analogs:
                continue
            distance = np.square(shapes[candidates] - shapes[t]).sum(axis=1)
            cutoff[t] = np.quantile(distance, params.analog_percentile)
            matched = outcomes[distance <= cutoff[t]]
            counts[t] = len(matched)
            if len(matched) >= params.min_analogs:
                lower[t], upper[t] = np.quantile(matched, [0.25, 0.75])
        out["distance_cutoff"] = cutoff
        out["analog_count"] = counts
        out["outcome_lower"] = lower
        out["outcome_upper"] = upper
        out["entry"] = np.where(lower > 0.0, 1, np.where(upper < 0.0, -1, 0))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeCycleParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for t in range(len(data)):
            if position:
                if t == exit_bar:
                    position = 0
                    continue
            elif entries[t] != 0:
                position = int(entries[t])
                exit_bar = t + params.hold_bars
            signal[t] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RangeCycleParams()
    strategy = GeneratedStrategy()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    t = np.arange(450, dtype=float)
    index = pd.date_range("2020-01-01", periods=len(t), freq="B")
    for side in (-1, 1):
        close = 100.0 * np.exp(side * 0.001 * t)
        width = close * 0.01 * (1.0 + 0.3 * np.sin(t / 3.0))
        data = pd.DataFrame({
            "open": close, "high": close + width / 2.0,
            "low": close - width / 2.0, "close": close, "volume": 1000000.0,
        }, index=index)
        result = strategy.generate_signals(
            data, strategy.indicators(data, params), ctx, params
        ).data
        s = result["signal"].to_numpy()
        starts = np.flatnonzero((s != 0) & np.r_[True, s[:-1] == 0])
        assert len(starts) > 0 and s[0] == 0
        for start in starts:
            if start + params.hold_bars < len(s):
                assert np.all(s[start:start + params.hold_bars] == side)
                assert s[start + params.hold_bars] == 0
        prefix = data.iloc[:400]
        earlier = strategy.generate_signals(
            prefix, strategy.indicators(prefix, params), ctx, params
        ).data
        assert earlier.equals(result.iloc[:400])
        assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
