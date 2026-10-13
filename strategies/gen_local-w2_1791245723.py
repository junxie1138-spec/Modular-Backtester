from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalStreakParams:
    lookback: int = 252
    atr_period: int = 14
    streak_length: int = 3
    seasonal_threshold: float = 0.5
    holding_bars: int = 7
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[SeasonalStreakParams]):
    strategy_id = "gen_local-w2_1791245723"

    @classmethod
    def params_type(cls):
        return SeasonalStreakParams

    def warmup_bars(self, params: SeasonalStreakParams) -> int:
        return int(max(params.lookback, params.atr_period, params.streak_length) + 1)

    def indicators(self, data: pd.DataFrame, params: SeasonalStreakParams) -> pd.DataFrame:
        if (params.lookback < 20 or params.atr_period < 1 or
                params.streak_length < 1 or params.holding_bars < 1 or
                not np.isfinite(params.stop_atr) or params.stop_atr <= 0 or
                not np.isfinite(params.seasonal_threshold) or params.seasonal_threshold < 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(close.diff()).fillna(0).astype(int)
        groups = out["direction"].ne(out["direction"].shift()).cumsum()
        out["streak"] = out.groupby(groups).cumcount().add(1).where(out["direction"].ne(0), 0)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["phase"] = np.minimum((data.index.day - 1) // 10, 2)
        out["seasonal_score"] = np.nan
        for phase in range(3):
            # Shift the entire masked series: today's return never trains today's seasonal estimate.
            history = out["return"].where(out["phase"].eq(phase)).shift(1)
            window = history.rolling(params.lookback, min_periods=20)
            out[f"phase_{phase}_mean"] = window.mean()
            out[f"phase_{phase}_std"] = window.std()
            out[f"phase_{phase}_count"] = history.rolling(params.lookback, min_periods=1).count()
            score = (out[f"phase_{phase}_mean"] /
                     out[f"phase_{phase}_std"].replace(0, np.nan) *
                     np.sqrt(out[f"phase_{phase}_count"]))
            out[f"phase_{phase}_score"] = score.replace([np.inf, -np.inf], np.nan)
            mask = out["phase"].eq(phase)
            out.loc[mask, "seasonal_score"] = out.loc[mask, f"phase_{phase}_score"]
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SeasonalStreakParams,
    ) -> SignalFrame:
        reversal = -indicators["direction"].where(
            indicators["streak"].ge(params.streak_length), 0
        )
        score = indicators["seasonal_score"]
        candidate = np.where(
            reversal.eq(1) & score.gt(params.seasonal_threshold), 1,
            np.where(reversal.eq(-1) & score.lt(-params.seasonal_threshold), -1, 0),
        )
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        side, age = 0, 0
        entry, distance = np.nan, np.nan
        warmup = max(self.warmup_bars(params), int(ctx.warmup_bars))
        for i in range(len(data)):
            if side:
                # The previous close's entry decision fills at this session's open.
                if np.isnan(entry):
                    entry = opens[i]
                age += 1
                stopped = (not np.isfinite(entry) or not np.isfinite(closes[i]) or
                           side * (closes[i] - entry) < -distance)
                if stopped or age >= params.holding_bars:
                    side, age = 0, 0
                    entry = np.nan
                signals[i] = side
                continue
            if (i >= warmup and candidate[i] and np.isfinite(closes[i]) and
                    np.isfinite(atrs[i]) and atrs[i] > 0):
                side = int(candidate[i])
                entry, distance, age = np.nan, params.stop_atr * atrs[i], 0
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SeasonalStreakParams(lookback=20, atr_period=1, streak_length=2, stop_atr=2.0)
    index = pd.bdate_range("2020-01-01", periods=40)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for side in (1, -1):
        data = pd.DataFrame(
            {"open": 100.0, "high": 111.0, "low": 89.0, "close": 100.0, "volume": 1000.0},
            index=index,
        )
        ind = pd.DataFrame(
            {"direction": 0, "streak": 0, "seasonal_score": 0.0, "atr": 1.0}, index=index
        )
        ind.iloc[22] = [-side, 2, side, 1.0]
        data.loc[index[24], "close"] = 100.0 + side * 10.0
        data.loc[index[25], "close"] = 100.0 - side
        data.loc[index[26], "close"] = 100.0 - side * 3.0
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].iloc[22] == 0
        assert (result["signal"].iloc[23:27] == side).all()
        assert result["signal"].iloc[27] == 0
        assert result["size"].gt(0).all()
        data["close"] = 100.0
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert (result["signal"].iloc[23:30] == side).all()
        assert result["signal"].iloc[30] == 0
        ind.loc[index[22], "seasonal_score"] = -side
        assert strategy.generate_signals(data, ind, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
