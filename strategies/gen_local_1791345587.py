from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecruitmentParams:
    participation_window: int = 6
    rank_window: int = 40
    atr_window: int = 14
    rank_threshold: float = 0.85
    breakeven_pct: float = 0.004
    trail_atr: float = 1.0
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RecruitmentParams]):
    strategy_id = "gen_local_1791345587"

    @classmethod
    def params_type(cls):
        return RecruitmentParams

    def warmup_bars(self, params: RecruitmentParams) -> int:
        return int(max(params.participation_window + params.rank_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecruitmentParams) -> pd.DataFrame:
        if (min(params.participation_window, params.rank_window, params.atr_window) < 2
                or not 0.0 < params.rank_threshold < 1.0
                or params.breakeven_pct <= 0.0 or params.trail_atr <= 0.0
                or not 0.0 < params.min_size <= params.max_size <= 1.0):
            raise ValueError("Invalid recruitment parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        volume = data["volume"].astype(float).clip(lower=0.0).replace([np.inf, -np.inf], np.nan)
        eligible_volume = volume.where(out["return"].notna())
        out["total_volume"] = eligible_volume.rolling(params.participation_window).sum()
        out["rising_volume"] = (eligible_volume * (out["return"] > 0)).rolling(
            params.participation_window).sum()
        out["share"] = out["rising_volume"] / out["total_volume"].replace(0.0, np.nan)
        previous_share = out["share"].shift(1)
        out["capacity"] = 4.0 * previous_share * (1.0 - previous_share)
        out["flux"] = out["share"].diff() * out["capacity"]
        out["flux_rank"] = out["flux"].abs().rolling(params.rank_window).rank(pct=True)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecruitmentParams,
    ) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        values = indicators[["return", "share", "flux", "flux_rank", "atr"]].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        age = 0
        entry = stop = 0.0
        armed = False
        trade_size = params.min_size
        for i in range(n):
            ret, share, flux, rank, atr = values[i]
            if side:
                # Yesterday's decision fills at today's open; no future open is read.
                if age == 0:
                    entry = opens[i]
                    entry_atr = values[i - 1, 4]
                    if not np.isfinite(entry) or entry <= 0.0:
                        side = 0
                        continue
                    stop = side * entry - params.trail_atr * entry_atr
                    armed = False
                age += 1
                # Stops are confirmed at the close and exit at the next open.
                signed_close = side * closes[i]
                exit_now = (not np.isfinite(signed_close) or closes[i] <= 0.0
                            or not np.isfinite(atr) or atr <= 0.0
                            or signed_close <= stop)
                if not exit_now:
                    if signed_close - side * entry >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed stop only rises: long stops rise, short stops fall.
                        stop = max(stop, side * entry, signed_close - params.trail_atr * atr)
                    exit_now = signed_close <= stop or age >= 2
                if exit_now:
                    side = 0
                    age = 0
                    continue
                raw[i] = side
                sizes[i] = trade_size
                continue
            if not np.isfinite(values[i]).all() or atr <= 0.0:
                continue
            if rank < params.rank_threshold or flux == 0.0:
                continue
            candidate = 1 if flux > 0.0 else -1
            if candidate * ret <= 0.0 or candidate * (share - 0.5) <= 0.0:
                continue
            side = candidate
            age = 0
            strength = np.clip((rank - params.rank_threshold) / (1.0 - params.rank_threshold), 0.0, 1.0)
            trade_size = params.min_size + (params.max_size - params.min_size) * strength
            raw[i] = side
            sizes[i] = trade_size
        df = pd.DataFrame({"signal": raw, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RecruitmentParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 98.0,
        "close": [100.0, 101.0, 100.2, 100.0, 100.0, 99.0, 99.8, 100.0],
        "volume": 1000.0,
    }, index=index)
    features = pd.DataFrame({
        "return": [0.01, 0.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
        "share": [0.6, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5],
        "flux": [0.2, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0],
        "flux_rank": [0.95, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "atr": 1.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert params.min_size < result["size"].iloc[1] < result["size"].iloc[5]
    stopped = data.copy()
    stopped.loc[index[1], "close"] = 98.0
    assert strategy.generate_signals(stopped, features, ctx, params).data["signal"].iloc[2] == 0
    prefix = strategy.generate_signals(data.iloc[:5], features.iloc[:5], ctx, params).data
    pd.testing.assert_frame_equal(result.iloc[:5], prefix)
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0, volume=0.0)
    empty_signal = strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data
    assert empty_signal["signal"].eq(0).all() and empty_signal["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
