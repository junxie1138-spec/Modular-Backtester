from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 32
    energy_threshold: float = 0.45
    autocorr_threshold: float = 0.10
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791243428"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(200, params.energy_window + 16))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 2 or params.holding_bars < 1 or params.stop_atr <= 0:
            raise ValueError("Window, holding period, and stop multiplier must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["drift"] = out["return"].rolling(8, min_periods=8).mean()
        out["autocorr"] = out["return"].rolling(
            params.energy_window, min_periods=params.energy_window
        ).corr(out["return"].shift(1))
        for scale in (2, 4, 8):
            block = out["return"].rolling(scale, min_periods=scale).mean()
            out[f"haar_{scale}"] = np.sqrt(scale / 2.0) * (block - block.shift(scale))
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        total = out["energy_2"] + out["energy_4"] + out["energy_8"]
        out["coarse_share"] = out["energy_8"] / total.where(total > 0)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        needed = ["ma200", "drift", "autocorr", "haar_8", "coarse_share", "atr"]
        ready = np.isfinite(indicators[needed]).all(axis=1)
        eligible = (
            ready
            & (indicators["atr"] > 0)
            & (indicators["coarse_share"] >= params.energy_threshold)
            & (indicators["autocorr"] >= params.autocorr_threshold)
        )
        long_entry = (
            eligible
            & (data["close"] > indicators["ma200"])
            & (indicators["drift"] > 0)
            & (indicators["haar_8"] > 0)
        )
        short_entry = (
            eligible
            & (data["close"] < indicators["ma200"])
            & (indicators["drift"] < 0)
            & (indicators["haar_8"] < 0)
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        side = 0
        pending_entry = False
        entry_price = 0.0
        entry_atr = 0.0
        age = 0
        for i in range(len(data)):
            if pending_entry:
                # The preceding close requested entry at this bar's open.
                entry_price = opens[i]
                pending_entry = False
                age = 0
                if not np.isfinite(entry_price) or entry_price <= 0:
                    side = 0
            if side:
                age += 1
                # Freeze ATR at the decision preceding entry; never trail the stop.
                stopped = np.isfinite(closes[i]) and (
                    side * (closes[i] - entry_price) < -params.stop_atr * entry_atr
                )
                if stopped or age >= params.holding_bars:
                    side = 0
                targets[i] = side
                continue
            if candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(candidates[i])
                entry_atr = atr[i]
                pending_entry = True
            targets[i] = side
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=4)
    for side in (1, -1):
        data = pd.DataFrame(
            {"open": [100.0] * 4,
             "close": [100.0, 100.0 + side, 100.0 - 5.0 * side, 100.0]},
            index=index,
        )
        derived = pd.DataFrame(
            {"ma200": 100.0 - 10.0 * side, "drift": float(side),
             "autocorr": 0.8, "haar_8": float(side),
             "coarse_share": 0.9, "atr": 2.0}, index=index,
        )
        result = strategy.generate_signals(data, derived, None, WaveletParams()).data
        assert result["signal"].tolist() == [0, side, side, 0]
        assert (result["size"] > 0).all()
    data["close"] = 100.0
    result = strategy.generate_signals(
        data, derived, None, WaveletParams(holding_bars=2)
    ).data
    assert result["signal"].tolist() == [0, -1, -1, 0]
