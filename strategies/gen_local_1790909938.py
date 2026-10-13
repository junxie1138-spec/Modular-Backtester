from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    baseline_bars: int = 30
    energy_bars: int = 3
    atr_bars: int = 14
    energy_trigger: float = 1.4
    share_jump: float = 0.12
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790909938"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(8 + params.energy_bars + params.baseline_bars,
                       8 + params.energy_bars + 4, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if min(params.baseline_bars, params.energy_bars, params.atr_bars, params.max_hold) < 1:
            raise ValueError("Window lengths and max_hold must be positive.")
        thresholds = [params.energy_trigger, params.share_jump, params.trail_atr]
        if not np.isfinite(thresholds).all() or min(thresholds) <= 0 or params.share_jump > 1:
            raise ValueError("Thresholds must be finite and positive; share_jump must not exceed one.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        # Causal, overlapping Haar details: adjacent equal-length blocks.
        for scale in (2, 4, 8):
            half = scale // 2
            out[f"mean_{half}"] = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"detail_{scale}"] = (np.sqrt(scale) / 2.0) * (
                out[f"mean_{half}"] - out[f"mean_{half}"].shift(half))
            out[f"energy_{scale}"] = out[f"detail_{scale}"].pow(2).rolling(
                params.energy_bars, min_periods=params.energy_bars).mean()
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        out["baseline_energy"] = out["total_energy"].shift(1).rolling(
            params.baseline_bars, min_periods=params.baseline_bars).mean()
        out["shock"] = out["total_energy"] / out["baseline_energy"].where(out["baseline_energy"] > 0)
        out["coarse_share"] = out["energy_8"] / out["total_energy"].where(out["total_energy"] > 0)
        out["share_change"] = out["coarse_share"] - out["coarse_share"].shift(3)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["strength"] = (0.25 + 0.70 * (
            out["share_change"] / params.share_jump * out["shock"] / params.energy_trigger - 1.0
        ).clip(0.0, 1.0)).fillna(0.25)
        out["entry"] = (
            (out["share_change"] > params.share_jump)
            & (out["share_change"].shift(1) <= params.share_jump)
            & (out["shock"] >= params.energy_trigger)
            & (out["detail_8"] > 0)
            & (out["detail_4"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strengths = indicators["strength"].to_numpy(dtype=float)
        active = False
        age = 0
        high_water = stop = -np.inf
        exposure = 0.25
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if valid:
                    # First update is on the entry-fill bar, excluding the decision bar.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or close <= stop or age >= params.max_hold:
                    active = False
            elif valid and entries[i] and np.isfinite(strengths[i]):
                active = True
                age = 0
                high_water = stop = -np.inf
                # The simulator sizes entries only: preserve entry conviction until exit.
                exposure = float(np.clip(strengths[i], 0.25, 0.95))
            if active:
                signals[i] = 1
                sizes[i] = exposure
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=96, freq="D")
    close = np.full(96, 100.0)
    close[52:54] = [104.0, 102.0]
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                         "close": close, "volume": 1000000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    ind["entry"] = False
    ind.loc[index[[50, 60]], "entry"] = True
    ind["strength"] = 0.8
    ind["atr"] = 1.0
    ind.loc[index[53:55], "atr"] = 4.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[50] == 0 and result["signal"].iloc[51] == 1
    assert result["signal"].iloc[54] == 1 and result["signal"].iloc[55] == 0
    assert result["signal"].iloc[70] == 1 and result["signal"].iloc[71] == 0
    assert result["size"].iloc[51] == 0.8 and (result["size"] > 0).all()
    assert set(result["signal"].unique()) <= {0, 1}
    short = strategy.indicators(data.iloc[:75], params)
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(short, full.iloc[:75])
