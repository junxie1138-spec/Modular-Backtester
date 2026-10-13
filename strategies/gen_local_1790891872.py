from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    innovation_threshold: float = 0.10
    trail_atr: float = 0.80


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790891872"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        # Twenty gap observations need a prior close, plus one regime lag.
        return 22

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        previous = close.shift(1)
        gap = opening.div(previous).sub(1)
        session = close.div(opening).sub(1)
        tr = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1).where(previous.notna())
        atr = tr.rolling(14, min_periods=14).mean().where(lambda x: x > 0)
        # Six-point least-squares latent slope; sum of squared weights is 17.5.
        slope = sum((2.5 - lag) * close.shift(lag) for lag in range(6)) / 17.5
        innovation = slope.diff().div(atr)
        coupling = gap.rolling(20, min_periods=20).corr(session).shift(1)
        return pd.DataFrame(
            {"gap": gap, "session": session, "true_range": tr, "atr": atr,
             "slope": slope, "innovation": innovation, "coupling": coupling},
            index=data.index,
        ).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if not (np.isfinite(params.innovation_threshold)
                and params.innovation_threshold > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Both parameters must be finite and positive.")
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        innovation = indicators["innovation"].to_numpy(dtype=float)
        coupling = indicators["coupling"].to_numpy(dtype=float)
        valid = (np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
                 & np.isfinite(atr) & (atr > 0)
                 & np.isfinite(innovation) & np.isfinite(coupling))
        regime = np.where(coupling > 0.15, 1, np.where(coupling < -0.15, -1, 0))
        candidate = np.where(
            valid & (np.abs(innovation) >= params.innovation_threshold),
            np.sign(np.nan_to_num(innovation, nan=0.0)) * regime, 0,
        ).astype(int)
        decisions = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry = peak = floor = 0.0
        armed = False
        warmup = max(self.warmup_bars(params), int(ctx.warmup_bars))
        for i in range(len(data)):
            if i < warmup:
                continue
            if side:
                if not valid[i]:
                    side = 0
                    continue
                opening, high, low, close = prices[i]
                if age == 0:
                    # The previous close's decision enters at this bar's open.
                    entry = opening
                    peak = side * entry
                    floor = peak - 1.5 * atr[i - 1]
                    armed = False
                age += 1
                favorable = side * (high if side > 0 else low)
                adverse = side * (low if side > 0 else high)
                hit_existing_stop = adverse <= floor
                peak = max(peak, favorable)
                if peak - side * entry >= 0.004 * entry:
                    armed = True
                if armed:
                    # Signed-price stops only rise; short price stops only fall.
                    floor = max(floor, side * entry, peak - params.trail_atr * atr[i])
                # Test the old stop intrabar and the newly raised stop at close.
                # New high/low ordering is unknown; do not assume an intrabar fill.
                if hit_existing_stop or side * close <= floor or age >= 2:
                    side = 0
                    continue
                decisions[i] = side
            elif valid[i] and candidate[i]:
                side = int(candidate[i])
                age = 0
                decisions[i] = side
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=27)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8,
         "close": 100.0, "volume": 1000.0}, index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=22)
    indicators = pd.DataFrame(
        {"atr": 1.0, "innovation": 0.0, "coupling": 0.5}, index=index,
    )
    for side in (1, -1):
        indicators.loc[index[22], "innovation"] = side * 0.2
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].iloc[22:26].tolist() == [0, side, side, 0]
        assert (result["size"] > 0).all()
        stopped = data.copy()
        if side > 0:
            stopped.loc[index[23], ["high", "close"]] = [101.0, 100.1]
        else:
            stopped.loc[index[23], ["low", "close"]] = [99.0, 99.9]
        result = strategy.generate_signals(stopped, indicators, ctx, params).data
        assert result["signal"].iloc[22:26].tolist() == [0, side, 0, 0]
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
