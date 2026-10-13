from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureCapacityParams:
    area_threshold: float = 0.2
    daily_vol_target: float = 0.008
    profit_target: float = 0.006
    max_hold_bars: int = 2

    def __post_init__(self):
        values = [self.area_threshold, self.daily_vol_target, self.profit_target]
        if not np.isfinite(values).all() or not 0 < self.area_threshold < 1:
            raise ValueError("Invalid area threshold or nonfinite parameter")
        if self.daily_vol_target <= 0 or self.profit_target <= 0:
            raise ValueError("Volatility and profit targets must be positive")
        if self.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two bars")


class GeneratedStrategy(BaseStrategy[SignatureCapacityParams]):
    strategy_id = "gen_local_1791304721"

    @classmethod
    def params_type(cls):
        return SignatureCapacityParams

    def warmup_bars(self, params: SignatureCapacityParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: SignatureCapacityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        out["return"] = np.log(close).diff()
        out["volume_baseline"] = volume.rolling(4, min_periods=4).mean()
        out["volume_surprise"] = volume / out["volume_baseline"] - 1.0
        r = out["return"]
        v = out["volume_surprise"]
        # Antisymmetric second-order signature of cumulative return/volume-surprise increments.
        out["signature_area"] = 0.5 * (
            r.shift(2) * v.shift(1) - v.shift(2) * r.shift(1)
            + r.shift(2) * v - v.shift(2) * r
            + r.shift(1) * v - v.shift(1) * r
        )
        out["area_scale"] = 0.5 * r.abs().rolling(3).sum() * v.abs().rolling(3).sum()
        out["area_score"] = out["signature_area"] / out["area_scale"].where(out["area_scale"] > 0)
        out["net_return"] = r.rolling(3, min_periods=3).sum()
        out["daily_vol"] = r.rolling(8, min_periods=8).std(ddof=0)
        out["entry_size"] = (params.daily_vol_target / out["daily_vol"].clip(lower=0.001)).clip(0.01, 0.95)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["ready"] = out[["area_score", "net_return", "daily_vol", "volume_surprise", "entry_size"]].notna().all(axis=1)
        return out.fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureCapacityParams,
    ) -> SignalFrame:
        score = indicators["area_score"]
        eligible = (
            indicators["ready"].astype(bool)
            & (score.abs() >= params.area_threshold)
            & (score * indicators["net_return"] > 0)
            & (indicators["volume_surprise"] > 0)
        )
        candidates = (-np.sign(score)).where(eligible, 0).astype(int).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_sizes = indicators["entry_size"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        position, held = 0, 0
        entry, exposure = np.nan, 0.01
        for i in range(len(data)):
            if position:
                # A prior close's entry decision fills at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                held += 1
                valid = np.isfinite(entry) and entry > 0 and np.isfinite(closes[i]) and closes[i] > 0
                if not valid:
                    position = 0
                elif position * (closes[i] / entry - 1.0) >= params.profit_target or held >= params.max_hold_bars:
                    position = 0
            elif candidates[i]:
                position = int(candidates[i])
                entry, held = np.nan, 0
                exposure = float(np.clip(entry_sizes[i], 0.01, 0.95))
            signals[i] = position
            sizes[i] = exposure
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SignatureCapacityParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    for side in (1, -1):
        data = pd.DataFrame({
            "open": 100.0, "high": 103.0, "low": 97.0,
            "close": [100.0, 100.0 + side * 2.0, 100.0, 100.0, 100.0, 100.0],
            "volume": 1000.0,
        }, index=index)
        features = pd.DataFrame({
            "area_score": -side * 0.5, "net_return": -side * 0.01,
            "volume_surprise": 0.5, "entry_size": 0.4, "ready": True,
        }, index=index)
        result = strategy.generate_signals(data, features, None, params).data
        assert result["signal"].tolist() == [0, side, 0, side, side, 0]
        assert (result["size"] > 0).all()
        actual = strategy.indicators(data, params)
        assert np.isfinite(actual.to_numpy(dtype=float)).all()
        assert not actual["ready"].any()
    assert strategy.warmup_bars(params) == 9


if __name__ == "__main__":
    _self_check()
