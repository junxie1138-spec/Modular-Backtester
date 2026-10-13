from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 5
    vol_window: int = 20
    area_threshold: float = 0.08
    gap_min: float = 0.001
    target_vol: float = 0.12
    vol_floor: float = 0.04
    max_size: float = 0.95
    profit_target: float = 0.008
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791256659"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(max(params.path_window, params.vol_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.path_window < 2 or params.vol_window < 2:
            raise ValueError("Lookback windows must be at least two bars")
        out = pd.DataFrame(index=data.index)
        out["log_high"] = np.log(data["high"].where(data["high"] > 0))
        out["log_low"] = np.log(data["low"].where(data["low"] > 0))
        out["high_step"] = out["log_high"].diff()
        out["low_step"] = out["log_low"].diff()
        out["edge_cross"] = (
            out["log_high"].shift(1) * out["log_low"]
            - out["log_low"].shift(1) * out["log_high"]
        )
        w = params.path_window
        # Antisymmetric second-order signature, translated to the window origin.
        out["signed_area"] = 0.5 * (
            out["edge_cross"].rolling(w, min_periods=w).sum()
            - out["log_high"].shift(w) * out["log_low"]
            + out["log_low"].shift(w) * out["log_high"]
        )
        out["high_travel"] = out["high_step"].abs().rolling(w, min_periods=w).sum()
        out["low_travel"] = out["low_step"].abs().rolling(w, min_periods=w).sum()
        out["area_scale"] = out["high_travel"] * out["low_travel"]
        out["low_first_area"] = -out["signed_area"] / out["area_scale"].where(out["area_scale"] > 1e-12)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["gap"] = data["open"] / data["close"].shift(1).where(data["close"].shift(1) > 0) - 1.0
        out["realized_vol"] = out["return"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["gap_recapture"] = ((out["gap"] <= -params.gap_min) & (out["return"] > 0)).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        if not 1 <= params.max_hold <= 2:
            raise ValueError("max_hold must be one or two bars")
        if not (params.profit_target > 0 and params.target_vol > 0 and params.vol_floor > 0 and 0 < params.max_size <= 1 and params.gap_min >= 0):
            raise ValueError("Invalid target, volatility, size, or gap parameters")
        ready = indicators[["low_first_area", "realized_vol", "gap", "return"]].notna().all(axis=1)
        enter = (
            ready
            & (indicators["low_first_area"] >= params.area_threshold)
            & indicators["gap_recapture"].eq(1)
        ).to_numpy(dtype=bool)
        exposure = (params.target_vol / indicators["realized_vol"].clip(lower=params.vol_floor)).clip(upper=params.max_size).fillna(params.max_size).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.max_size, dtype=float)
        active = False
        entry_price = np.nan
        entry_bar = -1
        frozen_size = params.max_size
        for i in range(len(data)):
            if active:
                # The previous close's order fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                take_profit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_stop = i - entry_bar + 1 >= params.max_hold
                sizes[i] = frozen_size
                if not valid_entry or take_profit or time_stop:
                    active = False
                    entry_price = np.nan
                else:
                    raw[i] = 1
            elif enter[i]:
                active = True
                entry_price = np.nan
                entry_bar = i + 1
                frozen_size = exposure[i]
                raw[i] = 1
                sizes[i] = frozen_size
        df = pd.DataFrame({"signal": raw, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"low_first_area": 0.5, "realized_vol": 0.2, "gap": -0.01, "return": 0.01, "gap_recapture": 1}, index=index)
    params = SignatureParams(profit_target=0.01)
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist()[:4] == [0, 1, 1, 0]
    assert np.isclose(result["size"].iloc[1], 0.6)
    data.loc[index[1], "close"] = 102.0
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist()[:3] == [0, 1, 0]
    ind["gap_recapture"] = 0
    assert GeneratedStrategy.generate_signals(data, ind, None, params).data["signal"].eq(0).all()
    ind["gap_recapture"] = 1
    ind["low_first_area"] = np.nan
    assert GeneratedStrategy.generate_signals(data, ind, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
