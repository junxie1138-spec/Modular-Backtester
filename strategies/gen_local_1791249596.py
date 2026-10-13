from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_threshold: float = 0.25
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791249596"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if not np.isfinite(params.area_threshold) or not 0.0 < params.area_threshold < 1.0:
            raise ValueError("area_threshold must lie between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        out["return"] = np.log(close).diff()
        out["downside_energy"] = out["return"].clip(upper=0.0).pow(2)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["annual_vol"] = out["return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        out["absolute_return_sum"] = out["return"].abs().rolling(5, min_periods=5).sum()
        out["downside_energy_sum"] = out["downside_energy"].rolling(5, min_periods=5).sum()
        out["net_return"] = out["return"].rolling(5, min_periods=5).sum()
        # Antisymmetric level-two signature of a five-increment path.
        area = pd.Series(0.0, index=data.index)
        for earlier in range(4, 0, -1):
            for later in range(earlier - 1, -1, -1):
                area += (
                    out["return"].shift(earlier) * out["downside_energy"].shift(later)
                    - out["downside_energy"].shift(earlier) * out["return"].shift(later)
                )
        out["signature_area"] = area
        out["area_scale"] = out["absolute_return_sum"] * out["downside_energy_sum"]
        out["normalized_area"] = area / out["area_scale"].where(out["area_scale"] > 0.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["normalized_area"] < -params.area_threshold)
            & (out["net_return"] > 0.0)
            & (out["return"] > 0.0)
            & (out["atr"] > 0.0)
            & (out["annual_vol"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        peak = np.nan
        stop = -np.inf
        prices = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            valid = np.isfinite(prices[i]) and prices[i] > 0.0 and np.isfinite(atr[i]) and atr[i] > 0.0
            if active:
                age += 1
                if not valid:
                    active = False
                else:
                    # First close after the entry decision belongs to the filled trade.
                    peak = prices[i] if not np.isfinite(peak) else max(peak, prices[i])
                    stop = max(stop, peak - params.stop_atr * atr[i])
                    if prices[i] <= stop or age >= 4:
                        active = False
            elif valid and entries[i]:
                active = True
                age = 0
                peak = np.nan
                stop = -np.inf
            signal[i] = int(active)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        exposure = (0.10 / indicators["annual_vol"].where(indicators["annual_vol"] > 0.0))
        exposure = exposure.replace([np.inf, -np.inf], np.nan).fillna(0.05).clip(0.05, 1.0)
        df["size"] = exposure.shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
        "annual_vol": [0.20] * 6,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    data = pd.DataFrame({"close": [100.0, 104.0, 103.0, 102.0, 101.0, 100.0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, SignatureParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert np.allclose(result["size"], [0.05, 0.5, 0.5, 0.5, 0.5, 0.5])
    data["close"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    indicators["atr"] = 1.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, SignatureParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]


if __name__ == "__main__":
    _self_check()
