from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDebtParams:
    gap_window: int = 4
    flow_window: int = 3
    atr_window: int = 14
    vol_window: int = 20
    debt_threshold: float = 0.8
    flow_threshold: float = 0.15
    target_vol: float = 0.12
    max_size: float = 0.95
    stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GapDebtParams]):
    strategy_id = "gen_local_1790971784"

    @classmethod
    def params_type(cls) -> type[GapDebtParams]:
        return GapDebtParams

    def warmup_bars(self, params: GapDebtParams) -> int:
        return int(max(params.gap_window + 1, params.flow_window,
                       params.atr_window + 1, params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapDebtParams) -> pd.DataFrame:
        p = params
        if min(p.gap_window, p.flow_window, p.atr_window) < 1 or p.vol_window < 2:
            raise ValueError("Invalid indicator windows")
        if not (3 <= p.hold_days <= 5 and 0 < p.max_size <= 1
                and p.target_vol > 0 and p.stop_atr > 0
                and p.debt_threshold > 0 and 0 <= p.flow_threshold < 1):
            raise ValueError("Invalid trading parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["range"] = data["high"] - data["low"]
        out["high_distance"] = (data["high"] - out["previous_close"]).abs()
        out["low_distance"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_distance", "low_distance"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["gap_sum"] = out["gap"].rolling(p.gap_window).sum()
        out["gap_debt"] = -out["gap_sum"] / out["prior_atr"].where(out["prior_atr"] > 0)
        out["intraday_pressure"] = ((data["close"] - data["open"]) /
                                     out["range"].where(out["range"] > 0)).clip(-1, 1)
        out.loc[out["range"].eq(0), "intraday_pressure"] = 0.0
        out["signed_volume"] = out["intraday_pressure"] * data["volume"].where(data["volume"] >= 0)
        out["flow_sum"] = out["signed_volume"].rolling(p.flow_window).sum()
        out["volume_sum"] = data["volume"].where(data["volume"] >= 0).rolling(p.flow_window).sum()
        out["flow"] = out["flow_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["annual_vol"] = out["return"].rolling(p.vol_window).std(ddof=1) * np.sqrt(252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapDebtParams) -> SignalFrame:
        p = params
        n = len(data)
        decision = np.zeros(n, dtype=int)
        sizes = np.full(n, float(p.max_size))
        eligible = ((indicators["gap_debt"] >= p.debt_threshold) &
                    (indicators["flow"] >= p.flow_threshold) &
                    (indicators["atr"] > 0) &
                    (indicators["annual_vol"] > 0))
        required = indicators[["gap_debt", "flow", "atr", "annual_vol"]]
        eligible &= np.isfinite(required).all(axis=1)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        vols = indicators["annual_vol"].to_numpy(dtype=float)
        entries = eligible.fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        entry_atr = 0.0
        stop_price = 0.0
        exposure = float(p.max_size)
        for i in range(n):
            if active:
                if i == entry_bar:
                    # The preceding close scheduled this entry at today's open.
                    stop_price = opens[i] - p.stop_atr * entry_atr
                invalid = (not np.isfinite(closes[i]) or closes[i] <= 0 or
                           not np.isfinite(stop_price) or
                           (i == entry_bar and opens[i] <= 0))
                if invalid or closes[i] < stop_price or i - entry_bar + 1 >= p.hold_days:
                    active = False
                else:
                    decision[i] = 1
                sizes[i] = exposure
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_atr = atrs[i]
                exposure = float(np.clip(p.target_vol / vols[i], 0.01, p.max_size))
                decision[i] = 1
                sizes[i] = exposure
        df = pd.DataFrame({"signal": decision, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(float(p.max_size)).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    p = GapDebtParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 96.0,
                         "close": [100., 100., 105., 99., 100., 100., 100.],
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"gap_debt": [1.] + [0.] * 6, "flow": 0.5,
                        "atr": 1.0, "annual_vol": 0.24}, index=index)
    run = lambda: GeneratedStrategy.generate_signals(None, data, ind, None, p).data
    result = run()
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.5)
    data.loc[index[2], "close"] = 97.0
    assert run()["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    ind["flow"] = -0.5
    assert run()["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
