from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_ratio: float = 1.4
    atr_multiple: float = 0.8


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790879334"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params):
        return 15

    def indicators(self, data, params):
        if not np.isfinite(params.energy_ratio) or params.energy_ratio <= 1:
            raise ValueError("energy_ratio must be finite and greater than one")
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("atr_multiple must be finite and positive")
        out = pd.DataFrame(index=data.index)
        log_close = np.log(data["close"].where(data["close"] > 0))
        out["log_close"] = log_close
        for width in (2, 4, 8):
            half = width // 2
            mean = log_close.rolling(half, min_periods=half).mean()
            out[f"haar_{width}"] = np.sqrt(width) * (mean - mean.shift(half)) / 2.0
            out[f"energy_{width}"] = out[f"haar_{width}"].pow(2)
        total = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        out["total_energy"] = total
        out["fine_share"] = out["energy_2"] / total.where(total > 0)
        out["prior_fine_share"] = out["fine_share"].shift(1)
        out["prior_coarse_energy"] = out["energy_8"].shift(1)
        out["prior_fine_energy"] = out["energy_2"].shift(1)
        previous_close = data["close"].shift(1)
        out["previous_close"] = previous_close
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        cascade = (
            (out["prior_coarse_energy"] > params.energy_ratio * out["prior_fine_energy"])
            & (out["energy_2"] > params.energy_ratio * out["energy_8"])
            & (out["fine_share"] > out["prior_fine_share"])
        )
        bullish = (out[["haar_2", "haar_4", "haar_8"]] > 0).all(axis=1)
        bearish = (out[["haar_2", "haar_4", "haar_8"]] < 0).all(axis=1)
        ready = np.isfinite(out).all(axis=1) & (out["atr"] > 0)
        out["entry"] = np.select(
            [ready & cascade & bullish, ready & cascade & bearish], [1, -1], default=0
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        raw = np.zeros(len(data), dtype=int)
        side = 0
        age = 0
        entry_price = 0.0
        stop = 0.0
        armed = False
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["entry"].fillna(0).to_numpy(dtype=int)
        for i, (open_price, high, low, close) in enumerate(prices):
            valid = np.isfinite([open_price, high, low, close, atr[i]]).all()
            valid = valid and min(open_price, high, low, close) > 0 and atr[i] > 0
            if not valid:
                side = 0
                age = 0
                continue
            if side:
                # The previous close's order fills at this bar's open.
                if age == 0:
                    entry_price = open_price
                    stop = side * entry_price - params.atr_multiple * atr[i - 1]
                    armed = False
                age += 1
                # Signed prices let both directions ratchet upward only.
                adverse = low if side == 1 else -high
                signed_close = side * close
                exited = adverse <= stop
                if not exited:
                    # Close-based activation avoids assuming high/low event order.
                    armed = armed or side * (close - entry_price) >= 0.004 * entry_price
                    if armed:
                        stop = max(stop, side * entry_price,
                                   signed_close - params.atr_multiple * atr[i])
                        exited = signed_close <= stop
                if exited or age >= 2:
                    side = 0
                    age = 0
                raw[i] = side
            elif candidates[i] in (-1, 1):
                side = int(candidates[i])
                age = 0
                raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=5)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    for side in (1, -1):
        data = pd.DataFrame({
            "open": 100.0, "high": 100.2, "low": 99.8,
            "close": 100.0, "volume": 1000.0,
        }, index=index)
        ind = pd.DataFrame({"atr": 1.0, "entry": [side, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, 0, 0]
        data.loc[index[1], "low" if side == 1 else "high"] = 99.0 if side == 1 else 101.0
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, side, 0, 0, 0]
        assert (result["size"] > 0).all()
