from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    energy_window: int = 4
    baseline_window: int = 40
    compression_ceiling: float = 0.65
    flow_window: int = 3
    flow_threshold: float = 0.25
    atr_window: int = 14
    initial_atr: float = 2.0
    breakeven_pct: float = 0.006
    trail_atr: float = 1.5
    max_hold: int = 4
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791281518"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(8 + params.energy_window + params.baseline_window + 1,
                       params.atr_window + 1, params.flow_window))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.energy_window, params.baseline_window, params.flow_window,
               params.atr_window, params.max_hold) < 1:
            raise ValueError("Lookbacks and max_hold must be positive.")
        if not (0 < params.compression_ceiling < 1 and
                0 <= params.flow_threshold < 1 and params.initial_atr > 0 and
                params.breakeven_pct > 0 and params.trail_atr > 0 and
                0 < params.size <= 1):
            raise ValueError("Invalid thresholds, stops, or size.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        # Causal, undecimated Haar details at spans 2, 4, and 8 sessions.
        ratios = []
        for span in (2, 4, 8):
            half = span // 2
            block = out["log_close"].rolling(half).mean()
            detail = np.sqrt(half / 2.0) * (block - block.shift(half))
            energy = detail.pow(2).rolling(params.energy_window).mean()
            baseline = energy.rolling(params.baseline_window).mean().shift(1)
            name = f"ratio_{span}"
            out[f"block_{span}"] = block
            out[f"detail_{span}"] = detail
            out[f"energy_{span}"] = energy
            out[f"baseline_{span}"] = baseline
            out[name] = energy / baseline.where(baseline > 0)
            ratios.append(name)
        out["compressed"] = (out[ratios].notna().all(axis=1) &
                             out[ratios].lt(params.compression_ceiling).all(axis=1))
        out["range"] = data["high"] - data["low"]
        out["close_location"] = ((2 * data["close"] - data["high"] - data["low"]) /
                                 out["range"].where(out["range"] > 0))
        volume = data["volume"].where(data["volume"] >= 0)
        out["valid_volume"] = volume.where(out["close_location"].notna())
        out["signed_volume"] = out["close_location"] * out["valid_volume"]
        out["volume_sum"] = out["valid_volume"].rolling(params.flow_window).sum()
        out["signed_sum"] = out["signed_volume"].rolling(params.flow_window).sum()
        out["flow"] = out["signed_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["range"], (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        # Two independent primitives must agree: spectral quiet AND buying pressure.
        out["entry"] = (out["compressed"] &
                        out["flow"].gt(params.flow_threshold) &
                        out["atr"].gt(0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        pending_atr = np.nan
        entry_index = -1
        entry_price = peak = stop = 0.0
        armed = False
        for i, (opening, high, low, close) in enumerate(prices):
            valid = np.isfinite(prices[i]).all() and min(prices[i]) > 0
            if not valid:
                entry_index = -1
                pending_atr = np.nan
                continue
            # A prior close's entry decision fills at this bar's observed open.
            if np.isfinite(pending_atr):
                entry_index = i
                entry_price = peak = opening
                stop = entry_price - params.initial_atr * pending_atr
                armed = False
                pending_atr = np.nan
            if entry_index >= 0:
                hit_existing_stop = low <= stop
                peak = max(peak, high)
                armed = armed or peak >= entry_price * (1 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                # New ratchets are checked at close, then govern subsequent bars.
                # Stop detection is a close decision; liquidation fills next open.
                exit_now = (hit_existing_stop or close <= stop or
                            i - entry_index + 1 >= params.max_hold)
                if exit_now:
                    entry_index = -1
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending_atr = atr[i]
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry": [True, False, False, False, False, False]},
                       index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    data.iloc[1, data.columns.get_indexer(["high", "close"])] = [102.0, 101.8]
    data.iloc[2, data.columns.get_indexer(["high", "low", "close"])] = [102.0, 100.6, 101.5]
    ind.iloc[2, ind.columns.get_loc("atr")] = 4.0
    data.iloc[3, data.columns.get_indexer(["high", "low", "close"])] = [102.0, 100.4, 101.0]
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
