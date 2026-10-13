from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 8
    rotation_lag: int = 3
    min_rotation: float = 0.06
    vol_window: int = 21
    target_vol: float = 0.12
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791320404"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(16 + params.energy_window + params.rotation_lag,
                       params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if (params.energy_window < 2 or params.rotation_lag < 1
                or params.vol_window < 2 or params.max_hold < 1
                or not np.isfinite(params.min_rotation)
                or params.min_rotation <= 0
                or not np.isfinite(params.target_vol)
                or params.target_vol <= 0
                or not np.isfinite(params.profit_target)
                or params.profit_target <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        for scale in (1, 4, 8):
            block = out["log_return"].rolling(scale, min_periods=scale).sum()
            out[f"haar_{scale}"] = (block - block.shift(scale)) / np.sqrt(2.0 * scale)
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        total = out["energy_1"] + out["energy_4"] + out["energy_8"]
        out["medium_share"] = out["energy_4"] / total.where(total > 0)
        out["slow_share"] = out["energy_8"] / total.where(total > 0)
        out["rotation"] = out["medium_share"] - out["medium_share"].shift(params.rotation_lag)
        pair = out["energy_4"] + out["energy_8"]
        out["balance"] = (
            2.0 * np.sqrt(out["energy_4"] * out["energy_8"])
            / pair.where(pair > 0)
        ).clip(0.0, 1.0)
        out["annual_vol"] = out["log_return"].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0) * np.sqrt(252.0)
        out["conviction"] = (
            (out["rotation"] / (2.0 * params.min_rotation)).clip(0.0, 1.0)
            * out["balance"]
        )
        out["target_size"] = (
            params.target_vol / out["annual_vol"].where(out["annual_vol"] > 0)
            * out["conviction"]
        ).clip(0.05, 0.95)
        required = ["haar_4", "haar_8", "rotation", "slow_share", "target_size"]
        out["ready"] = np.isfinite(out[required]).all(axis=1) & (out["annual_vol"] > 0)
        eligible = (
            out["ready"]
            & (out["rotation"] >= params.min_rotation)
            & (out["slow_share"] >= 0.15)
            & (out["haar_4"] * out["haar_8"] > 0)
        )
        out["entry"] = np.where(eligible, np.sign(out["haar_8"]), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.05, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        targets = indicators["target_size"].to_numpy(dtype=float)
        direction = 0
        held = 0
        entry_price = 0.0
        position_size = 0.05
        for i in range(n):
            if direction:
                # A decision on the preceding close fills at this bar's open.
                if held == 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        direction = 0
                        continue
                    entry_price = opens[i]
                held += 1
                gain = direction * (closes[i] / entry_price - 1.0)
                if (np.isfinite(gain) and gain >= params.profit_target) or held >= params.max_hold:
                    direction = 0
                    held = 0
                    # Exit decisions fill next bar; do not reenter on this close.
                    continue
            elif entries[i] and np.isfinite(targets[i]) and targets[i] > 0:
                direction = int(entries[i])
                held = 0
                position_size = float(np.clip(targets[i], 0.05, 0.95))
            signals[i] = direction
            if direction:
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveletParams(max_hold=3, profit_target=0.02)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"entry": [1, 0, 0, 0, 0, 0],
                            "target_size": 0.4}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result.loc[index[1], "size"] == 0.4
    for direction, winning_close in ((1, 103.0), (-1, 97.0)):
        derived.loc[index[0], "entry"] = direction
        data.loc[index[1], "close"] = winning_close
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, direction, 0, 0, 0, 0]
    flat = strategy.indicators(data.assign(close=100.0), params)
    assert np.isfinite(flat.to_numpy(dtype=float)).all()
    assert not flat["entry"].any()


if __name__ == "__main__":
    _self_check()
