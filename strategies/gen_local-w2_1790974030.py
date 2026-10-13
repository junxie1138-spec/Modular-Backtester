from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    lookback: int = 20
    atr_period: int = 14
    volume_multiple: float = 1.6
    impact_ceiling: float = 0.65
    min_drop_atr: float = 0.5
    breakeven_pct: float = 0.006
    trail_atr: float = 1.2
    initial_stop_atr: float = 1.5
    max_hold_bars: int = 2
    min_size: float = 0.25

    def __post_init__(self):
        if self.lookback < 2 or self.atr_period < 2:
            raise ValueError("Lookbacks must be at least two.")
        if self.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two sessions.")
        if not 0.0 < self.impact_ceiling < 1.0 or not 0.0 < self.min_size <= 1.0:
            raise ValueError("Invalid impact ceiling or minimum size.")
        values = (self.volume_multiple, self.min_drop_atr, self.breakeven_pct,
                  self.trail_atr, self.initial_stop_atr)
        if not all(np.isfinite(x) and x > 0.0 for x in values):
            raise ValueError("Thresholds and stop distances must be positive.")


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local-w2_1790974030"

    @classmethod
    def params_type(cls):
        return AbsorptionParams

    def warmup_bars(self, params: AbsorptionParams) -> int:
        return max(params.lookback + 2, params.atr_period + 1)

    def indicators(self, data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        change = data["close"].diff()
        volume = data["volume"].where(data["volume"] > 0.0)
        down = change < 0.0
        out["down_move"] = (-change).clip(lower=0.0)
        out["volume_reference"] = volume.rolling(
            params.lookback, min_periods=params.lookback
        ).mean().shift(1)
        out["relative_volume"] = volume / out["volume_reference"]
        out["sell_move_sum"] = out["down_move"].where(down, 0.0).rolling(
            params.lookback, min_periods=params.lookback
        ).sum().shift(1)
        out["sell_volume_sum"] = volume.where(down, 0.0).rolling(
            params.lookback, min_periods=params.lookback
        ).sum().shift(1)
        out["historical_impact"] = out["sell_move_sum"] / out["sell_volume_sum"].replace(0.0, np.nan)
        out["current_impact"] = out["down_move"] / volume
        out["impact_ratio"] = out["current_impact"] / out["historical_impact"].replace(0.0, np.nan)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["drop_atr"] = out["down_move"] / out["atr"].replace(0.0, np.nan)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["relative_volume"] >= params.volume_multiple)
            & (out["impact_ratio"] < params.impact_ceiling)
            & (out["drop_atr"] >= params.min_drop_atr)
            & (out["atr"] > 0.0)
        ).fillna(False)
        out["conviction"] = (
            (1.0 - out["impact_ratio"]).clip(lower=0.0)
            * out["relative_volume"] * out["drop_atr"] / 3.0
        ).clip(0.0, 1.0).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AbsorptionParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].fillna(0.0).to_numpy(dtype=float)
        active = False
        pending_entry = False
        armed = False
        entry_price = peak = stop = 0.0
        held = 0
        trade_size = params.min_size
        warmup = self.warmup_bars(params)
        for i in range(n):
            opening, high, low, close = prices[i]
            if not np.isfinite(prices[i]).all() or (prices[i] <= 0.0).any():
                active = pending_entry = False
                continue
            if pending_entry:
                # The opening price is observed only on the entry session.
                entry_price = opening
                peak = opening
                stop = opening - params.initial_stop_atr * atr[i - 1]
                held = 0
                armed = False
                active = True
                pending_entry = False
            if active:
                held += 1
                sizes[i] = trade_size
                # Test the previously established stop before using today's high.
                exit_now = low <= stop
                peak = max(peak, high)
                if peak >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atr[i]) and atr[i] > 0.0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                    exit_now = exit_now or close <= stop
                if exit_now or held >= params.max_hold_bars:
                    active = False
                    continue
                signal[i] = 1
            elif i >= warmup - 1 and entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                trade_size = params.min_size + (1.0 - params.min_size) * float(
                    np.clip(conviction[i], 0.0, 1.0)
                )
                sizes[i] = trade_size
                signal[i] = 1
                pending_entry = True
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = AbsorptionParams(lookback=2, atr_period=2)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100.0] * 8, "high": [100.2] * 8,
        "low": [99.7] * 8, "close": [100.0] * 8, "volume": [1000.0] * 8,
    }, index=index)
    ind = pd.DataFrame({
        "atr": 1.0, "entry": [False, False, False, True, False, False, False, False],
        "conviction": 0.8,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=4)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]
    assert np.isclose(result["size"].iloc[4], 0.85)
    data.loc[index[4], ["high", "close"]] = [101.0, 99.9]
    stopped = strategy.generate_signals(data, ind, ctx, params).data
    assert stopped["signal"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0]
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert not np.isinf(actual.select_dtypes(include="number").to_numpy()).any()
    quiet = strategy.generate_signals(data, actual, ctx, params).data
    assert set(quiet["signal"].unique()) <= {0, 1}
    assert (quiet["size"] > 0.0).all()
    prefix = strategy.generate_signals(data.iloc[:5], ind.iloc[:5], ctx, params).data
    pd.testing.assert_frame_equal(stopped.iloc[:5], prefix)


if __name__ == "__main__":
    _self_check()
