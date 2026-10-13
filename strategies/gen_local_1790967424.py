from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    area_window: int = 12
    vol_window: int = 20
    area_threshold: float = 0.18
    target_vol: float = 0.10
    profit_target: float = 0.025
    hold_bars: int = 8
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790967424"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.area_window, params.vol_window, 3) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.area_window < 3 or params.vol_window < 2:
            raise ValueError("Lookbacks must be at least three and two bars.")
        if not (0 < params.target_vol and 0 < params.max_size <= 1):
            raise ValueError("Volatility target and position cap must be positive.")
        if params.area_threshold < 0 or params.profit_target <= 0 or params.hold_bars < 1:
            raise ValueError("Invalid entry or exit parameters.")
        price = data["close"].where(data["close"] > 0)
        log_price = np.log(price)
        returns = log_price.diff()
        daily_vol = returns.rolling(params.vol_window).std(ddof=0)
        length = params.area_window
        displacement = log_price - log_price.shift(length)
        # Half the antisymmetric level-two signature of (normalized time, log price).
        area = (
            0.5 * (log_price + log_price.shift(length))
            - log_price.shift(1).rolling(length).mean()
            - displacement / (2.0 * length)
        )
        score = area / (daily_vol * np.sqrt(length)).where(daily_vol > 1e-12)
        strength = ((score - params.area_threshold) / 0.5).clip(0.15, 1.0)
        eligible = (
            (score > params.area_threshold)
            & (displacement < 0)
            & ((log_price - log_price.shift(3)) > 0)
        )
        out = pd.DataFrame(index=data.index)
        out["log_return"] = returns
        out["daily_vol"] = daily_vol
        out["displacement"] = displacement
        out["recent_return"] = log_price - log_price.shift(3)
        out["signature_area"] = area
        out["area_score"] = score
        out["annual_vol"] = daily_vol * np.sqrt(252.0)
        out["strength"] = strength
        out["eligible"] = eligible.fillna(False).astype(bool)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        entry_strength = 0.0
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].to_numpy(dtype=float)
        volatility = indicators["annual_vol"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0
                target_hit = (
                    valid_price and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if not valid_price or target_hit or expired:
                    active = False
                    continue
            elif (
                i >= self.warmup_bars(params) - 1
                and eligible[i] and np.isfinite(strength[i])
                and np.isfinite(volatility[i]) and volatility[i] > 0
            ):
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                entry_strength = float(strength[i])
            if active:
                signals[i] = 1
                vol = volatility[i]
                # ponytail: next-open proxy ignores fill costs; use fill-state hooks if available.
                risk_size = (
                    params.target_vol / vol
                    if np.isfinite(vol) and vol > 0 else 0.01
                )
                sizes[i] = float(np.clip(risk_size * entry_strength, 0.01, params.max_size))
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params(area_window=3, vol_window=2, hold_bars=3, profit_target=0.02)
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"eligible": False, "strength": 0.5, "annual_vol": 0.20}, index=index
    )
    indicators.loc[index[3], "eligible"] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=4)
    timed = strategy.generate_signals(data, indicators, ctx, params).data
    assert timed["signal"].tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0]
    assert np.isclose(timed["size"].iloc[4], 0.25)
    data.loc[index[4], "close"] = 103.0
    profit = strategy.generate_signals(data, indicators, ctx, params).data
    assert profit["signal"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert (profit["size"] > 0).all()
    assert strategy.indicators(data, params).index.equals(index)


if __name__ == "__main__":
    _self_check()
