from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    profit_target: float = 0.04
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791319403"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return 42

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if isinstance(params.holding_bars, bool) or not isinstance(params.holding_bars, int) or params.holding_bars < 1:
            raise ValueError("holding_bars must be a positive integer")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["lag_return"] = out["return"].shift(1)
        out["x"] = out["return"].fillna(0.0).cumsum()
        out["y"] = out["lag_return"].fillna(0.0).cumsum()
        out["cross_increment"] = (
            out["x"].shift(1) * out["lag_return"]
            - out["y"].shift(1) * out["return"]
        )
        out["dx"] = out["return"].rolling(12, min_periods=12).sum()
        out["dy"] = out["lag_return"].rolling(12, min_periods=12).sum()
        # Translate each twelve-increment path to its own origin.
        out["signature_area"] = 0.5 * (
            out["cross_increment"].rolling(12, min_periods=12).sum()
            - out["x"].shift(12) * out["dy"]
            + out["y"].shift(12) * out["dx"]
        )
        out["energy"] = (
            out["return"].pow(2) + out["lag_return"].pow(2)
        ).rolling(12, min_periods=12).sum()
        out["normalized_area"] = out["signature_area"] / out["energy"].where(out["energy"] > 0)
        out["autocorrelation"] = out["return"].rolling(40, min_periods=40).corr(out["lag_return"])
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            out["normalized_area"].gt(0)
            & out["normalized_area"].shift(1).le(0)
            & out["autocorrelation"].gt(0)
            & out["dx"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        raw = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = 0.0
        pending = False
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                # Yesterday's entry decision fills at today's open.
                entry_bar = i
                entry_price = opens[i]
            if entry_bar >= 0:
                hit_target = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.holding_bars
                if hit_target or expired:
                    entry_bar = -1
                else:
                    raw[i] = 1
                # An exit decision stays flat until at least the next close.
                continue
            if entries[i]:
                raw[i] = 1
                pending = True
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": [100.0] * 6, "close": [100.0, 100.0, 105.0, 100.0, 100.0, 100.0]},
        index=index,
    )
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, SignatureParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, None, SignatureParams(holding_bars=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
