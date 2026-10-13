from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletFluxParams:
    flux_threshold: float = 0.04
    profit_target: float = 0.04


class GeneratedStrategy(BaseStrategy[WaveletFluxParams]):
    strategy_id = "gen_local_1791341939"

    @classmethod
    def params_type(cls):
        return WaveletFluxParams

    def warmup_bars(self, params: WaveletFluxParams) -> int:
        return 40

    def indicators(self, data: pd.DataFrame, params: WaveletFluxParams) -> pd.DataFrame:
        if not np.isfinite(params.flux_threshold) or not 0 < params.flux_threshold < 1:
            raise ValueError("flux_threshold must be between zero and one")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for h in (2, 4, 8, 16):
            out[f"mean_{h}"] = out["log_close"].rolling(h, min_periods=h).mean()
            out[f"haar_{h}"] = np.sqrt(h / 2.0) * (
                out[f"mean_{h}"] - out[f"mean_{h}"].shift(h)
            )
            # Remove the h-cubed energy growth of a constant log-price slope.
            out[f"energy_{h}"] = (
                out[f"haar_{h}"].pow(2).rolling(5, min_periods=5).mean() / h**3
            )
        out["fine_energy"] = out["energy_2"] + out["energy_4"]
        out["coarse_energy"] = out["energy_8"] + out["energy_16"]
        out["total_energy"] = out["fine_energy"] + out["coarse_energy"]
        out["coarse_share"] = out["coarse_energy"] / out["total_energy"].where(
            out["total_energy"] > 0
        )
        out["flux"] = out["coarse_share"].diff(3)
        out["fine_change"] = out["fine_energy"].diff(3)
        out["coarse_change"] = out["coarse_energy"].diff(3)
        out["entry"] = (
            (out["flux"] > params.flux_threshold)
            & (out["flux"].shift(1) <= params.flux_threshold)
            & (out["fine_change"] < 0)
            & (out["coarse_change"] > 0)
            & (out["haar_8"] > 0)
            & (out["haar_16"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletFluxParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Gross target references the entry open before execution costs.
                    entry_price = opens[i]
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_stop = i - entry_bar + 1 >= 18
                if invalid_entry or target_hit or time_stop:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    raw[i] = 1
            elif i >= self.warmup_bars(params) - 1 and entries[i]:
                raw[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletFluxParams()
    index = pd.date_range("2020-01-01", periods=70, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    features = pd.DataFrame({"entry": False}, index=index)
    features.iloc[40, 0] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=40)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].iloc[40] == 0
    assert result["signal"].iloc[41:59].eq(1).all()
    assert result["signal"].iloc[59:].eq(0).all()
    data.iloc[41, data.columns.get_loc("open")] = 120.0
    data.iloc[42, data.columns.get_loc("close")] = 105.0
    data.iloc[43, data.columns.get_loc("close")] = 125.0
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].iloc[41:44].eq(1).all()
    assert result["signal"].iloc[44:].eq(0).all()
    assert result["size"].gt(0).all()
    data["close"] = 100.0 * np.exp(
        np.arange(70) * 0.001 + 0.01 * np.sin(np.arange(70) / 3.0)
    )
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:55], params)
    pd.testing.assert_frame_equal(full.iloc[:55], prefix)
    assert full["flux"].iloc[:38].isna().all()


if __name__ == "__main__":
    _self_check()
