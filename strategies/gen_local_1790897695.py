from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 8
    noise_window: int = 30
    threshold: float = 2.5
    drift: float = 0.25
    profit_target: float = 0.0075
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790897695"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.slope_window + params.noise_window + 3)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.slope_window < 2 or params.noise_window < 2
                or params.threshold <= 0 or params.drift < 0
                or params.profit_target <= 0 or params.max_hold not in (1, 2)):
            raise ValueError("Invalid slope, noise, threshold, drift, target, or holding period")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["slope"] = out["return"].rolling(params.slope_window).mean()
        out["innovation"] = out["slope"].diff()
        out["noise"] = out["innovation"].rolling(params.noise_window).std(ddof=0).shift(1)
        out["surprise"] = (out["innovation"] / out["noise"].where(out["noise"] > 1e-12)).replace([np.inf, -np.inf], np.nan)
        out["ma200"] = data["close"].rolling(200).mean()
        out["valid"] = out[["surprise", "ma200"]].notna().all(axis=1)
        out["up_step"] = (out["surprise"] - params.drift).where(out["valid"], 0.0)
        out["down_step"] = (-out["surprise"] - params.drift).where(out["valid"], 0.0)
        out["up_path"] = out["up_step"].cumsum()
        out["down_path"] = out["down_step"].cumsum()
        out["up_debt"] = out["up_path"] - out["up_path"].cummin().clip(upper=0.0)
        out["down_debt"] = out["down_path"] - out["down_path"].cummin().clip(upper=0.0)
        out["long_entry"] = (out["valid"] & (data["close"] > out["ma200"])
                             & (out["up_debt"] >= params.threshold)
                             & (out["up_debt"].shift(1).fillna(0.0) < params.threshold))
        out["short_entry"] = (out["valid"] & (data["close"] < out["ma200"])
                              & (out["down_debt"] >= params.threshold)
                              & (out["down_debt"].shift(1).fillna(0.0) < params.threshold))
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        side = 0
        entry_price = 0.0
        held = 0
        for i in range(len(data)):
            if side:
                # The previous close's entry decision fills at this bar's open.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i])
                gain = side * (closes[i] / entry_price - 1.0) if valid else 0.0
                if not valid or gain >= params.profit_target or held >= params.max_hold:
                    side = 0
                    held = 0
                    continue
            elif longs[i] or shorts[i]:
                side = 1 if longs[i] else -1
                held = 0
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    sample = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 98.0,
         "close": [100.0, 100.2, 100.4, 100.0, 100.0, 99.0, 99.0, 99.0],
         "volume": 1000.0}, index=index,
    )
    entries = pd.DataFrame(
        {"long_entry": [True, False, False, False, False, False, False, False],
         "short_entry": [False, False, False, False, True, False, False, False]},
        index=index,
    )
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(sample, entries, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, 0, 0]
    assert result["size"].gt(0).all()
