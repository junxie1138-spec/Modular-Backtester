from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 4
    percentile_bars: int = 126
    entry_percentile: float = 0.85
    min_efficiency: float = 0.25
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791206644"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(2 * params.path_bars + params.percentile_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.path_bars < 2 or params.percentile_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie between zero and one")
        if not 0.0 <= params.min_efficiency <= 1.0:
            raise ValueError("min_efficiency must lie between zero and one")
        if params.profit_target <= 0.0 or params.max_hold_bars not in (1, 2):
            raise ValueError("Require a positive target and a one- or two-bar holding limit")

        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0.0)
        closing = data["close"].where(data["close"] > 0.0)
        gap = np.log(opening / closing.shift(1))
        session = np.log(closing / opening)
        out["gap_return"] = gap
        out["session_return"] = session
        length = params.path_bars

        # Second-order antisymmetric signature: every ordered increment pair.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, length):
            cross = gap.shift(lag) * session - session.shift(lag) * gap
            area = area + 0.5 * cross.rolling(length - lag, min_periods=length - lag).sum()
        gap_activity = gap.abs().rolling(length, min_periods=length).sum()
        session_activity = session.abs().rolling(length, min_periods=length).sum()
        out["gap_activity"] = gap_activity
        out["session_activity"] = session_activity
        out["signature_area"] = area
        out["normalized_area"] = area.div((gap_activity * session_activity).replace(0.0, np.nan))
        out["previous_area"] = out["normalized_area"].shift(length)
        out["area_change"] = out["normalized_area"] - out["previous_area"]
        out["change_magnitude"] = out["area_change"].abs()
        out["change_threshold"] = out["change_magnitude"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile).shift(1)
        out["displacement"] = (gap + session).rolling(length, min_periods=length).sum()
        out["path_activity"] = gap_activity + session_activity
        out["efficiency"] = out["displacement"].abs().div(out["path_activity"].replace(0.0, np.nan))
        out["area_reversal"] = out["normalized_area"] * out["previous_area"] < 0.0
        out["entry_event"] = (
            out["area_reversal"]
            & (out["change_magnitude"] > out["change_threshold"])
            & (out["efficiency"] >= params.min_efficiency)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry_event"].fillna(False).to_numpy(dtype=bool)
        displacement = indicators["displacement"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1

        for i in range(len(data)):
            if position != 0:
                # Yesterday's entry decision fills at today's open.
                if i == entry_bar:
                    entry_price = opening[i]
                age = i - entry_bar + 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                gain = (
                    position * (closing[i] / entry_price - 1.0)
                    if valid_entry and np.isfinite(closing[i]) else np.nan
                )
                if not valid_entry or gain >= params.profit_target or age >= params.max_hold_bars:
                    position = 0
                    entry_price = np.nan
                    entry_bar = -1
            elif entries[i] and np.isfinite(displacement[i]) and displacement[i] != 0.0:
                position = 1 if displacement[i] > 0.0 else -1
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = position

        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
