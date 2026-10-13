from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_window: int = 6
    baseline_window: int = 40
    jump_z: float = 1.25
    hold_bars: int = 2

    def __post_init__(self):
        if self.area_window < 2 or self.baseline_window < 2:
            raise ValueError("Windows must be at least two bars.")
        if self.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        if not np.isfinite(self.jump_z) or self.jump_z <= 0:
            raise ValueError("jump_z must be finite and positive.")


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790898942"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return params.area_window + params.baseline_window + 2

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].div(data["close"].shift(1)).sub(1).replace([np.inf, -np.inf], np.nan)
        span = (data["high"] - data["low"]).replace(0, np.nan)
        out["pressure"] = ((2 * data["close"] - data["high"] - data["low"]) / span).clip(-1, 1)
        out.loc[data["high"].eq(data["low"]) & data["close"].notna(), "pressure"] = 0.0
        r, q = out["return"], out["pressure"]
        n = params.area_window
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature: sum over all ordered pairs.
        for lag in range(1, n):
            cross = r.shift(lag) * q - q.shift(lag) * r
            area = area + cross.rolling(n - lag, min_periods=n - lag).sum()
        out["area"] = 0.5 * area
        out["area_scale"] = r.abs().rolling(n).sum() * q.abs().rolling(n).sum()
        out["normalized_area"] = out["area"] / out["area_scale"].where(out["area_scale"] > 0)
        past = out["normalized_area"].shift(1).rolling(params.baseline_window)
        out["area_mean"] = past.mean()
        out["area_std"] = past.std(ddof=0)
        out["area_z"] = (out["normalized_area"] - out["area_mean"]) / out["area_std"].where(out["area_std"] > 1e-12)
        out["drift"] = r.rolling(2).sum()
        out["pressure_sum"] = q.rolling(2).sum()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        lead_change = indicators["area_z"].lt(-params.jump_z)
        longs = lead_change & indicators["drift"].gt(0) & indicators["pressure_sum"].gt(0)
        shorts = lead_change & indicators["drift"].lt(0) & indicators["pressure_sum"].lt(0)
        candidate = pd.Series(np.select([longs, shorts], [1, -1], default=0), index=data.index)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy(dtype=int)
        target = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        for i, direction in enumerate(confirmed):
            if position:
                if i - entry_bar == params.hold_bars:
                    position = 0
                # Expiry reserves this bar for flattening, even if another entry qualifies.
            elif direction:
                position, entry_bar = int(direction), i
            target[i] = position
        df = pd.DataFrame({"signal": target, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}, index=index)
    direction = pd.Series([0, 1, 1, 0, 0, -1, -1, 0, 0, 0], index=index)
    ind = pd.DataFrame({"area_z": -2.0, "drift": direction, "pressure_sum": direction}, index=index)
    strategy = GeneratedStrategy()
    params = SignatureParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, ctx, params).data["signal"].eq(0).all()
