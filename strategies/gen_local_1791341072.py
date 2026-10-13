from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_span: int = 12
    noise_window: int = 30
    innovation_threshold: float = 2.0
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791341072"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, 3 * params.trend_span + 2, params.noise_window + 2))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        returns = log_close.diff()
        latent_slope = returns.ewm(
            span=params.trend_span,
            adjust=False,
            min_periods=params.trend_span,
        ).mean()
        background_slope = returns.ewm(
            span=3 * params.trend_span,
            adjust=False,
            min_periods=3 * params.trend_span,
        ).mean()
        predicted_slope = latent_slope.shift(1)
        noise = returns.rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0).shift(1)
        innovation = ((returns - predicted_slope) / noise.where(noise > 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        valid = innovation.notna()
        queue_increment = (innovation.clip(-4.0, 4.0) - 0.25).where(valid, 0.0)
        cumulative = queue_increment.cumsum()
        surprise_queue = cumulative - cumulative.cummin()
        regime_ma = close.rolling(200, min_periods=200).mean()
        ready = (
            valid
            & latent_slope.notna()
            & background_slope.shift(1).notna()
            & regime_ma.notna()
        )
        entry = (
            ready
            & (close > regime_ma)
            & (background_slope.shift(1) <= 0.0)
            & (latent_slope > 0.0)
            & (surprise_queue >= params.innovation_threshold)
            & (surprise_queue.shift(1) < params.innovation_threshold)
        ).fillna(False)
        return pd.DataFrame(
            {
                "log_close": log_close,
                "returns": returns,
                "latent_slope": latent_slope,
                "background_slope": background_slope,
                "predicted_slope": predicted_slope,
                "noise": noise,
                "innovation": innovation,
                "queue_increment": queue_increment,
                "cumulative_surprise": cumulative,
                "surprise_queue": surprise_queue,
                "regime_ma": regime_ma,
                "ready": ready,
                "entry": entry,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held_bars = 0
        hold_limit = max(1, int(params.max_hold_bars))

        for i in range(len(data)):
            if active:
                # The preceding close's entry decision fills at this bar's open.
                if held_bars == 0:
                    entry_price = opens[i]
                held_bars += 1
                target_hit = (
                    np.isfinite(entry_price)
                    and entry_price > 0.0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                # Close-observed exits fill at the following open.
                if target_hit or held_bars >= hold_limit:
                    active = False
                    held_bars = 0
                    entry_price = np.nan
            elif entries[i]:
                active = True
                held_bars = 0
            signals[i] = int(active)

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [102.0] * 6,
            "low": [99.0] * 6,
            "close": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1000.0] * 6,
        },
        index=index,
    )
    indicators = pd.DataFrame(
        {"entry": [True, False, False, True, False, False]}, index=index
    )
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, StrategyParams()
    )
    assert result.data["signal"].tolist() == [0, 1, 0, 0, 1, 1]
    indicators["entry"] = [True, False, False, False, False, False]
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, StrategyParams()
    )
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
