from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalReleaseParams:
    pattern_window: int = 12
    entropy_gain: float = 0.03
    atr_window: int = 14
    stop_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[OrdinalReleaseParams]):
    strategy_id = "gen_local_1790929541"

    @classmethod
    def params_type(cls) -> type[OrdinalReleaseParams]:
        return OrdinalReleaseParams

    def warmup_bars(self, params: OrdinalReleaseParams) -> int:
        return int(max(params.pattern_window + 4, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalReleaseParams) -> pd.DataFrame:
        if params.pattern_window < 3 or params.atr_window < 1:
            raise ValueError("Lookback windows must be positive; pattern_window must be at least three.")
        if not np.isfinite(params.entropy_gain) or params.entropy_gain <= 0:
            raise ValueError("entropy_gain must be finite and positive.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_lag1"] = out["return"].shift(1)
        out["return_lag2"] = out["return"].shift(2)
        a, b, c = out["return_lag2"], out["return_lag1"], out["return"]
        out["ordinal"] = np.select([
            (a < b) & (b < c),
            (a < c) & (c < b),
            (b < a) & (a < c),
            (b < c) & (c < a),
            (c < a) & (a < b),
            (c < b) & (b < a),
        ], [0, 1, 2, 3, 4, 5], default=np.nan)
        # Equal returns have no strict ordinal pattern and suppress entries.
        for code in range(6):
            out[f"p{code}"] = (out["ordinal"] == code).astype(float).where(out["ordinal"].notna()).rolling(params.pattern_window).mean()
        probabilities = out[[f"p{code}" for code in range(6)]]
        out["entropy"] = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(axis=1, min_count=6) / np.log(6.0)
        out["entropy_change"] = out["entropy"].diff()
        out["new_up"] = ((out["ordinal"] == 0) & (out["p0"].shift(1) == 0)).astype(int)
        out["new_down"] = ((out["ordinal"] == 5) & (out["p5"].shift(1) == 0)).astype(int)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        valid = out["entropy_change"].ge(params.entropy_gain) & out["atr"].gt(0) & data["close"].gt(0)
        out["candidate"] = np.select([
            valid & out["new_up"].eq(1) & out["return"].gt(0),
            valid & out["new_down"].eq(1) & out["return"].lt(0),
        ], [1, -1], default=0).astype(int)
        out["impulse"] = (out["return"].abs() * previous_close / out["atr"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["confidence"] = ((out["entropy_change"] / (2.0 * params.entropy_gain)).clip(0, 1) * out["impulse"].clip(0, 1)).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalReleaseParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.2, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        confidence = indicators["confidence"].fillna(0).clip(0, 1).to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan
        held_size = 0.2
        for i in range(n):
            if side != 0:
                # The previous close's decision fills at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                bad_price = not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(closes[i])
                # Both the entry price and entry ATR stay fixed throughout the trade.
                stopped = not bad_price and side * (closes[i] - entry_price) < -params.stop_atr * entry_atr
                expired = i - entry_bar + 1 >= 2
                if bad_price or stopped or expired:
                    side = 0
            elif i + 1 < n and candidates[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
                entry_atr = atr[i]
                held_size = 0.2 + 0.75 * float(confidence[i])
            signals[i] = side
            sizes[i] = held_size if side != 0 else 0.2
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"open": [90., 100., 100., 100., 100., 100.], "close": [90., 100.5, 98., 100., 102., 100.]}, index=index)
    derived = pd.DataFrame({"atr": [1., 100., 100., 1., 100., 100.], "candidate": [1, 0, 0, -1, 0, 0], "confidence": [0.4, 0., 0., 0.8, 0., 0.]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(prices, derived, None, OrdinalReleaseParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    assert np.allclose(result["size"].iloc[[1, 2, 4]], [0.5, 0.5, 0.8])
    assert result["size"].gt(0).all()
    derived["candidate"] = [1, 0, 0, 0, 0, 0]
    prices["close"] = 100.5
    result = strategy.generate_signals(prices, derived, None, OrdinalReleaseParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
