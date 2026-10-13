from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalStretchParams:
    ma_bars: int = 40
    season_bars: int = 756
    min_samples: int = 12
    z_threshold: float = 1.0
    seasonal_edge: float = 0.006
    holding_bars: int = 18
    atr_bars: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    initial_stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[SeasonalStretchParams]):
    strategy_id = "gen_local_1790891347"

    @classmethod
    def params_type(cls):
        return SeasonalStretchParams

    def warmup_bars(self, params: SeasonalStretchParams) -> int:
        return int(max(params.ma_bars, params.atr_bars + 1,
                       params.season_bars + params.holding_bars))

    def indicators(self, data: pd.DataFrame, params: SeasonalStretchParams) -> pd.DataFrame:
        if min(params.ma_bars, params.season_bars, params.atr_bars,
               params.holding_bars) < 2 or params.min_samples < 1:
            raise ValueError("Lookbacks must be >= 2 and min_samples >= 1.")
        if not all(np.isfinite(x) and x > 0 for x in (
            params.z_threshold, params.seasonal_edge, params.breakeven_pct,
            params.trail_atr, params.initial_stop_atr
        )):
            raise ValueError("Thresholds and stop multiples must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["ma"] = close.rolling(params.ma_bars).mean()
        out["std"] = close.rolling(params.ma_bars).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].where(out["std"] > 0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["phase"] = 2 * (data.index.month - 1) + (data.index.day > 15).astype(int)
        out["origin_phase"] = out["phase"].shift(params.holding_bars)
        # A historical forecast becomes observable only at its horizon's end.
        out["matured_return"] = close / close.shift(params.holding_bars) - 1.0
        out["seasonal_return"] = np.nan
        out["seasonal_count"] = np.nan
        for phase in range(24):
            match = out["origin_phase"].eq(phase) & out["matured_return"].notna()
            count = match.astype(float).rolling(params.season_bars).sum()
            total = out["matured_return"].where(match, 0.0).rolling(params.season_bars).sum()
            estimate = total / count.where(count >= params.min_samples)
            current = out["phase"].eq(phase)
            out.loc[current, "seasonal_return"] = estimate.loc[current]
            out.loc[current, "seasonal_count"] = count.loc[current]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SeasonalStretchParams
    ) -> SignalFrame:
        z = indicators["z"].to_numpy(dtype=float)
        seasonal = indicators["seasonal_return"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        desired[(z <= -params.z_threshold) & (seasonal >= params.seasonal_edge)] = 1
        desired[(z >= params.z_threshold) & (seasonal <= -params.seasonal_edge)] = -1
        raw = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        entry = 0.0
        stop_floor = 0.0
        armed = False
        first = self.warmup_bars(params) - 1
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if position:
                if age == 0:
                    # Previous close's decision fills at this bar's open.
                    if not np.isfinite(opening[i]) or opening[i] <= 0 or not valid:
                        position = 0
                        continue
                    entry = opening[i]
                    stop_floor = -params.initial_stop_atr * atr[i - 1]
                age += 1
                if not valid:
                    position = 0
                    continue
                profit = position * (close[i] - entry)
                # Stops are close-confirmed; the flat signal fills next open.
                if profit <= stop_floor or age >= params.holding_bars:
                    position = 0
                    continue
                armed = armed or profit >= params.breakeven_pct * entry
                if armed:
                    # Signed profit floor only rises, for both long and short.
                    stop_floor = max(stop_floor, 0.0, profit - params.trail_atr * atr[i])
            elif i >= first and valid and desired[i]:
                position = int(desired[i])
                age = 0
                armed = False
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SeasonalStretchParams(ma_bars=2, season_bars=2, atr_bars=2, min_samples=1)
    index = pd.bdate_range("2020-01-01", periods=42)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    start = ctx.warmup_bars - 1
    for direction in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 105.0, "low": 95.0,
                             "close": 100.0, "volume": 1000.0}, index=index)
        data.iloc[start + 2:start + 5, data.columns.get_loc("close")] = (
            100.0 + direction * np.array([3.0, 4.0, 1.0])
        )
        ind = pd.DataFrame({"z": 0.0, "seasonal_return": direction * 0.01,
                            "atr": 1.0}, index=index)
        ind.iloc[start, ind.columns.get_loc("z")] = -direction * 2.0
        signals = strategy.generate_signals(data, ind, ctx, params).data["signal"]
        assert signals.iloc[start:start + 6].tolist() == [0, direction, direction, direction, direction, 0]
        ind["seasonal_return"] = -direction * 0.01
        assert not strategy.generate_signals(data, ind, ctx, params).data["signal"].any()
        ind["seasonal_return"] = direction * 0.01
        data["close"] = 100.0
        signals = strategy.generate_signals(data, ind, ctx, params).data["signal"]
        assert signals.iloc[start + params.holding_bars] == direction
        assert signals.iloc[start + params.holding_bars + 1] == 0
    computed = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:30], params)
    pd.testing.assert_frame_equal(computed.iloc[:30], prefix)
    assert not strategy.generate_signals(data, computed, ctx, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
