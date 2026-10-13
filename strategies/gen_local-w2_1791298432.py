from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAnchorParams:
    min_body_pct: float = 0.004
    anchor_tolerance: float = 0.25
    min_repair: float = 0.25
    profit_pct: float = 0.006
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GapAnchorParams]):
    strategy_id = "gen_local-w2_1791298432"

    @classmethod
    def params_type(cls):
        return GapAnchorParams

    @classmethod
    def warmup_bars(cls, params: GapAnchorParams) -> int:
        return 201

    def indicators(self, data: pd.DataFrame, params: GapAnchorParams) -> pd.DataFrame:
        prior_open = data["open"].shift(1)
        prior_close = data["close"].shift(1)
        prior_ma = data["close"].rolling(200, min_periods=200).mean().shift(1)
        body = prior_close - prior_open
        gap = data["open"] - prior_close
        body_pct = body.div(prior_open.replace(0, np.nan))
        anchor_error = (data["open"] - prior_open).abs().div(body.abs().replace(0, np.nan))
        repair = (data["close"] - data["open"]).div((-gap).replace(0, np.nan))
        regime = np.where(prior_close > prior_ma, 1, np.where(prior_close < prior_ma, -1, 0))
        return pd.DataFrame({
            "body_pct": body_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "gap": gap.fillna(0.0),
            "anchor_error": anchor_error.replace([np.inf, -np.inf], np.nan).fillna(np.inf),
            "repair": repair.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "regime": regime,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAnchorParams,
    ) -> SignalFrame:
        common = (
            (indicators["anchor_error"] <= params.anchor_tolerance)
            & (indicators["repair"] >= params.min_repair)
            & (indicators["repair"] <= 0.85)
        )
        long_entry = common & (indicators["regime"] == 1) & (indicators["body_pct"] >= params.min_body_pct) & (indicators["gap"] < 0)
        short_entry = common & (indicators["regime"] == -1) & (indicators["body_pct"] <= -params.min_body_pct) & (indicators["gap"] > 0)
        longs = long_entry.to_numpy()
        shorts = short_entry.to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        raw = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        entry_price = 0.0
        held_bars = 0

        for i in range(len(data)):
            if pending:
                position = pending
                pending = 0
                entry_price = float(opens[i])
                held_bars = 0
            if position:
                held_bars += 1
                gain = position * (float(closes[i]) / entry_price - 1.0) if entry_price > 0 else 0.0
                if gain >= params.profit_pct or held_bars >= params.max_hold:
                    position = 0
                else:
                    raw[i] = position
            elif longs[i] or shorts[i]:
                pending = 1 if longs[i] else -1
                raw[i] = pending

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
