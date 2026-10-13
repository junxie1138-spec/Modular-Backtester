from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveEnergyParams:
    energy_window: int = 24
    energy_share: float = 0.40
    autocorr_ceiling: float = -0.10
    atr_window: int = 14
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[WaveEnergyParams]):
    strategy_id = "gen_local_1790908739"

    @classmethod
    def params_type(cls) -> type[WaveEnergyParams]:
        return WaveEnergyParams

    @staticmethod
    def warmup_bars(params: WaveEnergyParams) -> int:
        return int(max(params.energy_window + 8, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveEnergyParams) -> pd.DataFrame:
        if params.energy_window < 3 or params.atr_window < 1:
            raise ValueError("Invalid rolling window")
        if not 0.0 < params.energy_share < 1.0:
            raise ValueError("energy_share must lie between zero and one")
        if not -1.0 < params.autocorr_ceiling < 0.0 or params.stop_atr <= 0.0:
            raise ValueError("Invalid autocorrelation ceiling or stop distance")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        r = out["return"]
        out["block2"] = r.rolling(2, min_periods=2).sum()
        out["block4"] = r.rolling(4, min_periods=4).sum()
        out["haar2"] = (r - r.shift(1)) / np.sqrt(2.0)
        out["haar4"] = (out["block2"] - out["block2"].shift(2)) / 2.0
        out["haar8"] = (out["block4"] - out["block4"].shift(4)) / np.sqrt(8.0)
        for scale in (2, 4, 8):
            out[f"energy{scale}"] = out[f"haar{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        total = out[["energy2", "energy4", "energy8"]].sum(axis=1, min_count=3)
        out["energy_share"] = out["energy4"] / total.where(total > 0.0)
        out["autocorr2"] = r.rolling(
            params.energy_window, min_periods=params.energy_window
        ).corr(r.shift(2))
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [(data["high"] - data["low"]),
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveEnergyParams,
    ) -> SignalFrame:
        ready = np.isfinite(indicators[["energy_share", "autocorr2", "block2", "atr"]]).all(axis=1)
        candidate = (
            ready
            & (indicators["energy_share"] >= params.energy_share)
            & (indicators["autocorr2"] <= params.autocorr_ceiling)
            & (indicators["block2"] < 0.0)
            & (indicators["atr"] > 0.0)
        ).fillna(False)
        confirmed = candidate & candidate.shift(1, fill_value=False)
        entry_event = confirmed & ~confirmed.shift(1, fill_value=False)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = entry_event.to_numpy(dtype=bool)
        desired = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            # A previous close's entry decision fills at this bar's open.
            if not active and i > 0 and desired[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(atr[i - 1]):
                    active = True
                    entry_bar = i
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                else:
                    continue
            if active:
                # Freeze the entry ATR and opening-price reference for the entire trade.
                stopped = not np.isfinite(closes[i]) or closes[i] < fixed_stop
                expired = i - entry_bar >= 1
                if stopped or expired:
                    active = False
                else:
                    desired[i] = 1
            elif entries[i]:
                desired[i] = 1
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 97.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"energy_share": [0.6, 0.6, 0.0, 0.0, 0.0, 0.0],
                        "autocorr2": -0.3, "block2": -0.01, "atr": 1.0}, index=index)
    params = WaveEnergyParams()
    normal = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert normal["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 98.0
    stopped = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert stopped["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    ind.loc[index[1], "energy_share"] = 0.0
    unconfirmed = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert unconfirmed["signal"].eq(0).all()
