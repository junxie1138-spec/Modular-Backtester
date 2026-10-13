from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 4
    vol_bars: int = 8
    area_threshold: float = 0.06
    target_daily_vol: float = 0.008


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791255009"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        if not (2 <= params.path_bars <= 9 and 2 <= params.vol_bars <= 9):
            raise ValueError("Lookbacks must be between 2 and 9.")
        if not (0 < params.area_threshold < 0.5):
            raise ValueError("area_threshold must be between 0 and 0.5.")
        if not (np.isfinite(params.target_daily_vol) and params.target_daily_vol > 0):
            raise ValueError("target_daily_vol must be finite and positive.")
        return int(max(params.path_bars, params.vol_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        r = np.log(data["close"].where(data["close"] > 0)).diff()
        v = np.log(data["volume"].where(data["volume"] > 0)).diff()
        r = r.replace([np.inf, -np.inf], np.nan)
        v = v.replace([np.inf, -np.inf], np.nan)
        n = params.path_bars
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature of (log-volume, log-price).
        # Older volume increments preceding price increments contribute positively.
        for older in range(n - 1, 0, -1):
            for newer in range(older - 1, -1, -1):
                area += 0.5 * (
                    v.shift(older) * r.shift(newer)
                    - r.shift(older) * v.shift(newer)
                )
        scale = v.abs().rolling(n).sum() * r.abs().rolling(n).sum()
        vol = r.rolling(params.vol_bars).std(ddof=0)
        ready = area.notna() & scale.gt(0) & vol.gt(0)
        score = (area / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)
        target_size = (params.target_daily_vol / vol.where(vol > 0)).clip(0.05, 1.0)
        out["price_return"] = r.fillna(0.0)
        out["volume_change"] = v.fillna(0.0)
        out["signature_area"] = area.fillna(0.0)
        out["area_scale"] = scale.fillna(0.0)
        out["area_score"] = score.fillna(0.0)
        out["daily_vol"] = vol.fillna(0.0)
        out["ready"] = ready.fillna(False)
        out["target_size"] = target_size.fillna(0.05)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        score = indicators["area_score"].to_numpy()
        ready = indicators["ready"].to_numpy(dtype=bool)
        sizes = indicators["target_size"].to_numpy()
        desired = np.where(
            ready & (score > params.area_threshold), 1,
            np.where(ready & (score < -params.area_threshold), -1, 0),
        )
        signals = np.zeros(len(data), dtype=int)
        allocations = np.full(len(data), 0.05, dtype=float)
        side = 0
        entry_size = 0.05
        for i, direction in enumerate(desired):
            if side != 0 and direction == -side:
                side = 0
            elif side == 0 and direction != 0:
                side = int(direction)
                entry_size = float(sizes[i])
            signals[i] = side
            allocations[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": allocations}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=24)
    t = np.arange(24, dtype=float)
    close = 100 * np.exp(0.002 * t + 0.01 * np.sin(t))
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000000 * np.exp(0.2 * np.cos(t)),
    }, index=index)
    ind = strategy.indicators(data, params)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert strategy.warmup_bars(params) == 9
    assert np.isfinite(ind.to_numpy(dtype=float)).all()
    assert result["signal"].isin([-1, 0, 1]).all() and result["size"].gt(0).all()
    prefix = strategy.indicators(data.iloc[:16], params)
    pd.testing.assert_frame_equal(ind.iloc[:16], prefix)
    controlled = ind.iloc[:8].copy()
    controlled["area_score"] = [0, 0.2, 0, -0.2, -0.2, 0, 0.2, 0.2]
    controlled["ready"] = True
    controlled["target_size"] = 0.5
    actual = strategy.generate_signals(data.iloc[:8], controlled, ctx, params).data
    assert actual["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert actual["size"].iloc[2] == 0.5


if __name__ == "__main__":
    _self_check()
