from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    slope_window: int = 12
    noise_window: int = 20
    regime_window: int = 30
    regime_threshold: float = 0.20
    entry_z: float = 0.80
    profit_target: float = 0.02
    max_hold: int = 8
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790883101"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.slope_window + params.noise_window + params.regime_window + 3)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.slope_window, params.noise_window, params.regime_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < params.regime_threshold < 1 and params.entry_z > 0):
            raise ValueError("Invalid regime or entry threshold.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        clock = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["slope"] = out["log_close"].rolling(params.slope_window).cov(
            clock, ddof=0
        ) / ((params.slope_window ** 2 - 1) / 12.0)
        out["innovation"] = out["slope"].diff()
        out["innovation_scale"] = out["innovation"].rolling(
            params.noise_window
        ).std(ddof=0).shift(1)
        out["innovation_z"] = out["innovation"] / out["innovation_scale"].where(
            out["innovation_scale"] > 1e-12
        )
        out["serial_correlation"] = out["innovation"].rolling(
            params.regime_window
        ).corr(out["innovation"].shift(1))
        out["regime"] = np.select(
            [out["serial_correlation"] >= params.regime_threshold,
             out["serial_correlation"] <= -params.regime_threshold],
            [1, -1], default=0
        ).astype(int)
        out["candidate"] = np.where(
            out["innovation_z"].abs() >= params.entry_z,
            np.sign(out["innovation_z"].fillna(0)) * out["regime"],
            0
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.max_hold < 1 or params.profit_target <= 0 or params.size <= 0:
            raise ValueError("Holding period, target, and size must be positive.")
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(
            candidate.eq(candidate.shift(1))
            & indicators["regime"].eq(indicators["regime"].shift(1)),
            0
        ).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction, held, entry_price = 0, 0, np.nan
        for i in range(len(data)):
            if direction:
                # Capture the entry open only after the decision bar has passed.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                gain = direction * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or held >= params.max_hold:
                    direction, held, entry_price = 0, 0, np.nan
                raw[i] = direction
            elif confirmed[i] != 0:
                direction = int(confirmed[i])
                held = 0
                raw[i] = direction
        df = pd.DataFrame({"signal": raw, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(max_hold=2)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for side in (1, -1):
        indicators = pd.DataFrame(
            {"candidate": [side, side, 0, 0, 0, 0], "regime": 1},
            index=index
        )
        flat = strategy.generate_signals(data, indicators, ctx, params).data
        assert flat["signal"].tolist() == [0, 0, side, side, 0, 0]
        profitable = data.copy()
        profitable.iloc[2, profitable.columns.get_loc("close")] = 100 * (1 + side * 0.03)
        target = strategy.generate_signals(profitable, indicators, ctx, params).data
        assert target["signal"].tolist() == [0, 0, side, 0, 0, 0]
        indicators.iloc[1, indicators.columns.get_loc("regime")] = -1
        assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
