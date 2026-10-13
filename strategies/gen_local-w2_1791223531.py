from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletSpringParams:
    short_scale: int = 8
    mid_scale: int = 16
    long_scale: int = 32
    energy_window: int = 32
    energy_multiple: float = 1.25
    agreement_floor: float = 0.20
    anchor_span: int = 21
    pullback_window: int = 5
    pullback_atr: float = 0.50
    atr_window: int = 14
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[WaveletSpringParams]):
    strategy_id = "gen_local-w2_1791223531"

    @classmethod
    def params_type(cls):
        return WaveletSpringParams

    def warmup_bars(self, params: WaveletSpringParams) -> int:
        return int(params.long_scale + params.energy_window)

    def indicators(self, data: pd.DataFrame, params: WaveletSpringParams) -> pd.DataFrame:
        scales = (params.short_scale, params.mid_scale, params.long_scale)
        if any(scale < 2 or scale % 2 for scale in scales) or not scales[0] < scales[1] < scales[2]:
            raise ValueError("Wavelet scales must be ascending positive even integers")
        if params.energy_window < 2 or params.anchor_span < 2 or params.pullback_window < 1 or params.atr_window < 2:
            raise ValueError("Indicator windows are invalid")
        if not np.isfinite(params.energy_multiple) or params.energy_multiple <= 0:
            raise ValueError("energy_multiple must be finite and positive")
        if not np.isfinite(params.agreement_floor) or not 0 <= params.agreement_floor <= 1:
            raise ValueError("agreement_floor must be between zero and one")
        if not np.isfinite(params.pullback_atr) or params.pullback_atr <= 0:
            raise ValueError("pullback_atr must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")

        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        out = pd.DataFrame(index=data.index)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        energy_columns = []
        slope_columns = []
        for scale in scales:
            half = scale // 2
            recent_mean = log_close.rolling(half, min_periods=half).mean()
            slope_name = f"wave_slope_{scale}"
            energy_name = f"wave_energy_{scale}"
            out[slope_name] = (recent_mean - recent_mean.shift(half)) / half
            out[energy_name] = out[slope_name].pow(2)
            slope_columns.append(slope_name)
            energy_columns.append(energy_name)

        energies = out[energy_columns]
        out["wave_energy"] = energies.sum(axis=1, min_count=len(energy_columns))
        out["wave_agreement"] = energies.min(axis=1).div(energies.max(axis=1).replace(0.0, np.nan))
        out["energy_baseline"] = out["wave_energy"].rolling(
            params.energy_window, min_periods=params.energy_window
        ).median().shift(1)

        anchor = close.ewm(span=params.anchor_span, adjust=False).mean()
        out["spring_tension"] = (close - anchor).div(out["atr"].replace(0.0, np.nan))
        out["prior_stretch"] = out["spring_tension"].shift(1).rolling(
            params.pullback_window, min_periods=params.pullback_window
        ).min()

        wavelet_agrees = (
            out[slope_columns].gt(0).all(axis=1)
            & out["wave_agreement"].ge(params.agreement_floor)
            & out["energy_baseline"].gt(0)
            & out["wave_energy"].gt(params.energy_multiple * out["energy_baseline"])
        )
        spring_releases = (
            out["prior_stretch"].le(-params.pullback_atr)
            & out["spring_tension"].ge(0)
            & out["spring_tension"].gt(out["spring_tension"].shift(1))
        )
        out["entry"] = (wavelet_agrees & spring_releases).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletSpringParams,
    ) -> SignalFrame:
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")

        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        in_position = False
        highest_close = -np.inf
        trailing_stop = -np.inf

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                signals[i] = int(in_position)
                continue

            if in_position:
                highest_close = max(highest_close, close)
                trailing_stop = max(trailing_stop, highest_close - params.trail_atr * atr)
                if close <= trailing_stop:
                    in_position = False
                    highest_close = -np.inf
                    trailing_stop = -np.inf
            elif entries[i] == 1:
                in_position = True
                highest_close = close
                trailing_stop = close - params.trail_atr * atr

            signals[i] = int(in_position)

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    close = [100.0, 101.0, 103.0, 104.0, 103.0, 102.0, 101.0, 101.0]
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, 0, 0, 0, 0, 0]}, index=index)
    result = GeneratedStrategy().generate_signals(data, derived, None, WaveletSpringParams(trail_atr=1.5)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
