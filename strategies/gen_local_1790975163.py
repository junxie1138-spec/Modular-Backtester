from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletRecoilParams:
    energy_ratio: float = 1.5
    profit_target: float = 0.0075


class GeneratedStrategy(BaseStrategy[WaveletRecoilParams]):
    strategy_id = "gen_local_1790975163"

    @classmethod
    def params_type(cls):
        return WaveletRecoilParams

    def warmup_bars(self, params: WaveletRecoilParams) -> int:
        # Eight-return Haar support plus a 24-bar energy window.
        return 32

    def indicators(self, data: pd.DataFrame, params: WaveletRecoilParams) -> pd.DataFrame:
        if not all(np.isfinite(x) and x > 0 for x in (params.energy_ratio, params.profit_target)):
            raise ValueError("Parameters must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = np.log(close.where(close > 0)).diff().replace([np.inf, -np.inf], np.nan)
        r = out["return"]
        for scale in (2, 4, 8):
            half = scale // 2
            recent = r.rolling(half, min_periods=half).sum()
            out[f"detail_{scale}"] = (recent - recent.shift(half)) / np.sqrt(scale)
            out[f"energy_{scale}"] = out[f"detail_{scale}"].pow(2).rolling(24, min_periods=24).mean()
        out["autocorrelation"] = r.rolling(24, min_periods=24).corr(r.shift(1)).replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        recoil = (
            ready
            & (out["autocorrelation"] < -0.10)
            & (out["energy_2"] > params.energy_ratio * (out["energy_4"] + out["energy_8"]))
            & (out["detail_2"] * r > 0)
        )
        out["entry"] = (-np.sign(out["detail_2"]).where(recoil, 0.0)).fillna(0.0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletRecoilParams) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # The previous close's instruction fills at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                held_bars = i - entry_bar + 1
                if not valid_entry or gain >= params.profit_target or held_bars >= 2:
                    position = 0
                    entry_price = np.nan
                    signals[i] = 0
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 97.0, "close": [100.0, 101.0, 102.0, 100.0, 99.0, 98.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=32)
    target = strategy.generate_signals(data, indicators, ctx, WaveletRecoilParams(profit_target=0.005)).data
    assert target["signal"].tolist() == [0, 1, 0, 0, -1, 0]
    expiry = strategy.generate_signals(data, indicators, ctx, WaveletRecoilParams(profit_target=0.5)).data
    assert expiry["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert (expiry["size"] > 0).all()
