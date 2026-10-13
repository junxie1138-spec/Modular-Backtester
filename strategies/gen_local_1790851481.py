from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRegimeParams:
    entropy_cutoff: float = 0.85
    activation_pct: float = 0.006


class GeneratedStrategy(BaseStrategy[OrdinalRegimeParams]):
    strategy_id = "gen_local_1790851481"

    @classmethod
    def params_type(cls) -> type[OrdinalRegimeParams]:
        return OrdinalRegimeParams

    def warmup_bars(self, params: OrdinalRegimeParams) -> int:
        # Three returns, 18 pattern observations, then five entropy observations.
        return 26

    def indicators(self, data: pd.DataFrame, params: OrdinalRegimeParams) -> pd.DataFrame:
        if not 0.0 < params.entropy_cutoff < 1.0:
            raise ValueError("entropy_cutoff must be between zero and one")
        if not np.isfinite(params.activation_pct) or params.activation_pct <= 0.0:
            raise ValueError("activation_pct must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["return"] = (close / previous - 1.0).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        a, b, c = out["return"].shift(2), out["return"].shift(1), out["return"]
        valid = a.notna() & b.notna() & c.notna()
        # Strict comparisons break ties by chronological order.
        out["pattern"] = ((a > b).astype(int) + 2 * (a > c).astype(int)
                          + 4 * (b > c).astype(int)).where(valid)
        probability_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            name = f"probability_{code}"
            out[name] = out["pattern"].eq(code).astype(float).where(valid).rolling(
                18, min_periods=18
            ).mean()
            probability_columns.append(name)
        probabilities = out[probability_columns]
        out["entropy"] = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(
            axis=1, min_count=6
        ) / np.log(6.0)
        out["entropy_baseline"] = out["entropy"].rolling(5, min_periods=5).mean()
        out["impulse"] = (close - previous) / out["atr"].where(out["atr"] > 0.0)
        coordinated = (out["entropy"] < params.entropy_cutoff) & (
            out["entropy"] < out["entropy_baseline"]
        )
        fragmented = (out["entropy"] >= params.entropy_cutoff) & (
            out["entropy"] > out["entropy_baseline"]
        )
        impulse_direction = np.sign(out["impulse"].fillna(0.0))
        out["setup"] = np.where(coordinated, impulse_direction,
                                np.where(fragmented, -impulse_direction, 0))
        out["setup"] = out["setup"].where(out["impulse"].abs() >= 0.6, 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalRegimeParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        direction = 0
        entry_bar = -1
        entry = 0.0
        stop = -np.inf
        best = -np.inf
        armed = False
        for i in range(len(data)):
            if direction == 0:
                if setups[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                    direction = int(setups[i])
                    entry_bar = i + 1
                    raw[i] = direction
                continue
            opening, high, low, close = prices[i]
            if not np.isfinite(prices[i]).all() or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                direction = 0
                continue
            if i == entry_bar:
                # The preceding close's decision fills at this session's open.
                entry = opening
                if entry <= 0.0:
                    direction = 0
                    continue
                stop = direction * entry - 1.25 * atr[i - 1]
                best = direction * entry
                armed = False
            # Signed prices let both long and short protective stops ratchet upward.
            adverse = low if direction == 1 else high
            if direction * adverse <= stop:
                direction = 0
                continue
            favorable = high if direction == 1 else low
            best = max(best, direction * favorable)
            if best - direction * entry >= entry * params.activation_pct:
                armed = True
            if armed:
                stop = max(stop, direction * entry, best - 0.9 * atr[i])
            # Newly raised stops are checked at the close; no assumed intrabar order.
            # A zero decision exits at the following open, as required by the contract.
            if (armed and direction * close <= stop) or i - entry_bar + 1 >= 2:
                direction = 0
                continue
            raw[i] = direction
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = OrdinalRegimeParams()
    index = pd.date_range("2020-01-01", periods=4)
    for direction in (1, -1):
        for excursion, expected in ((2.0, [0, direction, 0, 0]),
                                    (0.3, [0, direction, direction, 0])):
            sample = pd.DataFrame({
                "open": [100.0] * 4,
                "high": [100.1, 100.0 + (excursion if direction == 1 else 0.2), 100.3, 100.3],
                "low": [99.9, 100.0 - (excursion if direction == -1 else 0.2), 99.7, 99.7],
                "close": [100.0, 100.0 + direction * 0.2, 100.0, 100.0],
                "volume": [1000.0] * 4,
            }, index=index)
            derived = pd.DataFrame({"atr": 1.0, "setup": [direction, 0, 0, 0]}, index=index)
            result = strategy.generate_signals(sample, derived, None, params).data
            assert result["signal"].tolist() == expected
            assert (result["size"] > 0.0).all()
