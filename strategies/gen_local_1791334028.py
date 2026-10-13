from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_threshold: float = 0.20
    target_vol: float = 0.12

    def __post_init__(self):
        if not np.isfinite(self.area_threshold) or not 0 < self.area_threshold < 0.5:
            raise ValueError("area_threshold must be between zero and 0.5")
        if not np.isfinite(self.target_vol) or self.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791334028"

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return 21

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        out["volume_change"] = np.log(data["volume"].where(data["volume"] > 0)).diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        x = pd.Series(0.0, index=data.index)
        y = x.copy()
        area = x.copy()
        price_length = x.copy()
        volume_length = x.copy()
        # Six increments define a local, translation-invariant second-order signature.
        for lag in range(5, -1, -1):
            dx = out["return"].shift(lag)
            dy = out["volume_change"].shift(lag)
            area = area + 0.5 * (x * dy - y * dx)
            x = x + dx
            y = y + dy
            price_length = price_length + dx.abs()
            volume_length = volume_length + dy.abs()
        denominator = (price_length * volume_length).replace(0.0, np.nan)
        out["signature_area"] = area / denominator
        out["displacement"] = x
        out["annual_vol"] = out["return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        out["target_size"] = (params.target_vol / out["annual_vol"].replace(0.0, np.nan)).clip(0.05, 0.95)
        eligible = (
            out["signature_area"].abs().ge(params.area_threshold)
            & (out["signature_area"] * out["displacement"]).gt(0)
            & out["annual_vol"].gt(0)
            & out["target_size"].notna()
        )
        out["candidate"] = (-np.sign(out["signature_area"])).where(eligible, 0).fillna(0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        proposed_sizes = indicators["target_size"].to_numpy(dtype=float)
        direction = 0
        entry_bar = -1
        entry_size = 1.0
        for i in range(len(data)):
            if direction != 0:
                if i - entry_bar == 4:
                    direction = 0
                else:
                    signals[i] = direction
                    sizes[i] = entry_size
            elif candidates[i] != 0 and np.isfinite(proposed_sizes[i]) and proposed_sizes[i] > 0:
                direction = int(candidates[i])
                entry_bar = i
                entry_size = float(proposed_sizes[i])
                signals[i] = direction
                sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    bars = pd.DataFrame(index=index)
    features = pd.DataFrame(
        {"candidate": [1] + [-1] * 11, "target_size": np.linspace(0.2, 0.8, 12)},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(bars, features, None, SignatureParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, -1]
    assert np.allclose(result["size"].iloc[1:5], 0.2)
    assert np.allclose(result["size"].iloc[6:10], features["target_size"].iloc[5])
    assert result["size"].gt(0).all()
