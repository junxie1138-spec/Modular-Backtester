from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BranchingParams:
    pattern_window: int = 18
    percentile_window: int = 126
    drawdown_window: int = 20
    atr_window: int = 14
    stress_quantile: float = 0.8
    stop_atr: float = 1.25
    holding_sessions: int = 2


class GeneratedStrategy(BaseStrategy[BranchingParams]):
    strategy_id = "gen_local-w2_1791342720"

    @classmethod
    def params_type(cls):
        return BranchingParams

    def warmup_bars(self, params: BranchingParams) -> int:
        if min(params.pattern_window, params.percentile_window,
               params.drawdown_window, params.atr_window) < 2:
            raise ValueError("Lookback windows must be at least two.")
        if not 0.0 < params.stress_quantile < 1.0:
            raise ValueError("stress_quantile must lie strictly between zero and one.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite.")
        if params.holding_sessions not in (1, 2):
            raise ValueError("holding_sessions must be one or two.")
        return int(max(max(params.pattern_window + 4, params.drawdown_window)
                       + params.percentile_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: BranchingParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        a, b, c = out["return"].shift(2), out["return"].shift(1), out["return"]
        # Strict inversions give a stable chronological ordering for ties.
        valid = a.notna() & b.notna() & c.notna()
        out["pattern"] = (4 * (a > b).astype(int) + 2 * (a > c).astype(int)
                          + (b > c).astype(int)).where(valid)
        out["transition"] = 8 * out["pattern"].shift(1) + out["pattern"]
        patterns = (0, 1, 3, 4, 6, 7)
        source_columns = []
        edge_columns = []
        for origin in patterns:
            columns = []
            for destination in patterns:
                name = f"edge_{origin}_{destination}"
                present = (out["transition"] == 8 * origin + destination).astype(float)
                present = present.where(out["transition"].notna())
                out[name] = present.rolling(
                    params.pattern_window, min_periods=params.pattern_window).sum()
                columns.append(name)
                edge_columns.append(name)
            name = f"successors_{origin}"
            out[name] = out[columns].gt(0).sum(axis=1).astype(float).where(
                out[columns].notna().all(axis=1))
            source_columns.append(name)
        out["active_sources"] = out[source_columns].gt(0).sum(axis=1).astype(float)
        ready = out[edge_columns].notna().all(axis=1)
        out["branching"] = (out[source_columns].sum(axis=1)
                            / out["active_sources"].replace(0, np.nan)).where(ready)
        out["branching_threshold"] = out["branching"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.stress_quantile)
        out["peak"] = data["close"].rolling(
            params.drawdown_window, min_periods=params.drawdown_window).max()
        out["drawdown"] = (1.0 - data["close"] / out["peak"].replace(0, np.nan)).clip(lower=0)
        out["drawdown_threshold"] = out["drawdown"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.stress_quantile)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1).where(
            data[["high", "low", "close"]].notna().all(axis=1))
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["entry"] = (
            ready
            & (out["branching"] >= out["branching_threshold"])
            & (out["drawdown"].shift(1) >= out["drawdown_threshold"].shift(1))
            & (out["drawdown"].shift(1) > 0)
            & (out["drawdown"] > 0)
            & (data["close"] > previous_close)
            & (data["close"] > data["open"])
            & np.isfinite(out["atr"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: BranchingParams
    ) -> SignalFrame:
        self.warmup_bars(params)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active, age, stop = False, 0, np.nan
        for i in range(len(data)):
            # Yesterday's decision fills at today's open; ATR stays frozen.
            if not active and i > 0 and raw[i - 1] == 1:
                if (np.isfinite(opens[i]) and opens[i] > 0
                        and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    active, age = True, 0
                    stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                age += 1
                exit_now = (not np.isfinite(closes[i])
                            or closes[i] < stop
                            or age >= params.holding_sessions)
                raw[i] = 0 if exit_now else 1
                if exit_now:
                    active = False
                continue
            if entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 112.0, "low": 89.0,
        "close": [100.0, 110.0, 99.0, 100.0, 90.0, 100.0, 100.0],
        "volume": 1000.0
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0,
        "entry": [True, False, False, True, False, False, False]
    }, index=index)
    strategy, params = GeneratedStrategy(), BranchingParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, 1, 0, 0]
    params.holding_sessions = 1
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 0, 0, 1, 0, 0]
    assert (signals["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
