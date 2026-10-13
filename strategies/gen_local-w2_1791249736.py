from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParticipationParams:
    energy_window: int = 5
    coarse_share: float = 0.65
    participation_window: int = 6
    participation_floor: float = 0.62
    atr_window: int = 14
    atr_multiple: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[WaveletParticipationParams]):
    strategy_id = "gen_local-w2_1791249736"

    @classmethod
    def params_type(cls):
        return WaveletParticipationParams

    @staticmethod
    def warmup_bars(params: WaveletParticipationParams) -> int:
        windows = (params.energy_window, params.participation_window,
                   params.atr_window, params.max_hold)
        if any(not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in windows):
            raise ValueError("Windows and max_hold must be positive integers.")
        if not (0 < params.coarse_share < 1 and
                0 < params.participation_floor < 1 and
                np.isfinite(params.atr_multiple) and params.atr_multiple > 0):
            raise ValueError("Invalid thresholds or ATR multiple.")
        return int(max(16 + params.energy_window - 1,
                       params.participation_window, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParticipationParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for scale in (2, 4, 8, 16):
            half = scale // 2
            recent = out["log_close"].rolling(half, min_periods=half).sum()
            # Causal, undecimated Haar coefficient: adjacent trailing blocks.
            out[f"haar_{scale}"] = (recent - recent.shift(half)) / np.sqrt(scale)
            # Scale adjustment removes the mechanical random-walk energy slope.
            out[f"energy_{scale}"] = (
                out[f"haar_{scale}"].pow(2)
                .rolling(params.energy_window, min_periods=params.energy_window).mean()
                / scale ** 2
            )
        out["coarse_energy"] = out["energy_8"] + out["energy_16"]
        out["fine_energy"] = out["energy_2"] + out["energy_4"]
        out["coarse_share"] = out["coarse_energy"] / (
            out["coarse_energy"] + out["fine_energy"]
        ).replace(0, np.nan)
        out["wavelet_regime"] = (
            (out["coarse_share"] >= params.coarse_share)
            & (out["haar_8"] > 0) & (out["haar_16"] > 0)
        ).fillna(False)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0)
        out["up_volume"] = out["valid_volume"] * (data["close"] > data["open"])
        out["total_volume"] = out["valid_volume"].rolling(
            params.participation_window, min_periods=params.participation_window
        ).sum()
        out["total_up_volume"] = out["up_volume"].rolling(
            params.participation_window, min_periods=params.participation_window
        ).sum()
        out["participation"] = out["total_up_volume"] / out["total_volume"].replace(0, np.nan)
        out["volume_regime"] = (
            out["participation"] >= params.participation_floor
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParticipationParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = (
            indicators["wavelet_regime"].fillna(False)
            & indicators["volume_regime"].fillna(False)
        ).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        age = 0
        for i, (close, atr) in enumerate(zip(closes, atrs)):
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if not valid:
                active = False
                high_water, stop, age = np.nan, -np.inf, 0
                continue
            if active:
                # The first close after the entry decision is the first in-trade close.
                high_water = close if age == 0 else max(high_water, close)
                stop = max(stop, high_water - params.atr_multiple * atr)
                age += 1
                if close <= stop or age >= params.max_hold:
                    active = False
            elif entries[i]:
                active = True
                high_water, stop, age = np.nan, -np.inf, 0
            targets[i] = int(active)
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = WaveletParticipationParams(atr_multiple=2.0)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 100., 104., 103., 101., 100.]}, index=index)
    derived = pd.DataFrame({
        "wavelet_regime": [True, False, False, False, False, False],
        "volume_regime": [True, False, False, False, False, False],
        "atr": [1., 1., 1., 10., 10., 1.],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
    for primitive in ("wavelet_regime", "volume_regime"):
        blocked = derived.copy()
        blocked[primitive] = False
        assert not GeneratedStrategy.generate_signals(data, blocked, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
