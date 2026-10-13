from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    entropy_window: int = 24
    entropy_gap: float = 0.12
    volatility_window: int = 20
    atr_window: int = 14
    target_volatility: float = 0.12
    atr_multiple: float = 1.5


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local_1791212523"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params):
        return int(max(params.entropy_window + 2,
                       params.volatility_window + 1, params.atr_window + 1, 3))

    def indicators(self, data, params):
        if (params.entropy_window < 12 or params.volatility_window < 2
                or params.atr_window < 1 or not 0 < params.entropy_gap < 1
                or not np.isfinite(params.target_volatility)
                or params.target_volatility <= 0
                or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0):
            raise ValueError("Invalid ordinal strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["pressure"] = ((data.close - data.open)
                           / (data.high - data.low).replace(0, np.nan))
        triples = pd.concat([out.pressure.shift(2),
                             out.pressure.shift(1), out.pressure], axis=1)
        # Stable sorting breaks equal-pressure ties by chronological order.
        order = np.argsort(triples.to_numpy(), axis=1, kind="stable")
        out["pattern"] = pd.Series(order @ np.array([9, 3, 1]),
                                   index=data.index, dtype=float)
        valid = np.isfinite(triples.to_numpy()).all(axis=1)
        out.loc[~valid, "pattern"] = np.nan
        for label, direction in (("up", 1), ("down", -1)):
            side = direction * out.pressure > 0
            out[label + "_count"] = side.astype(float).where(valid).rolling(
                params.entropy_window, min_periods=params.entropy_window).sum()
            terms = []
            for code in (5, 7, 11, 15, 19, 21):
                name = label + "_" + str(code)
                out[name + "_count"] = ((out.pattern == code) & side).astype(
                    float).where(valid).rolling(
                    params.entropy_window, min_periods=params.entropy_window).sum()
                out[name + "_probability"] = (out[name + "_count"]
                    / out[label + "_count"].replace(0, np.nan))
                probability = out[name + "_probability"]
                terms.append(-probability * np.log(probability.clip(lower=1e-12)))
            out[label + "_entropy"] = sum(terms) / np.log(6.0)
            out.loc[out[label + "_count"] < 6, label + "_entropy"] = np.nan
        out["entropy_balance"] = out.down_entropy - out.up_entropy
        out["two_day_change"] = data.close.diff(2)
        out["return"] = data.close.pct_change(fill_method=None)
        out["annual_volatility"] = out["return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0) * np.sqrt(252.0)
        previous = data.close.shift(1)
        out["true_range"] = pd.concat([
            data.high - data.low, (data.high - previous).abs(),
            (data.low - previous).abs()], axis=1).max(axis=1)
        out.loc[previous.isna(), "true_range"] = np.nan
        out["atr"] = out.true_range.rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["exposure"] = (params.target_volatility
            / out.annual_volatility.replace(0, np.nan)).clip(lower=0.01, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        ready = (np.isfinite(indicators.entropy_balance)
                 & np.isfinite(indicators.two_day_change)
                 & np.isfinite(indicators.exposure)
                 & np.isfinite(indicators.atr) & (indicators.atr > 0))
        raw = pd.Series(0, index=data.index, dtype=int)
        raw.loc[ready & (indicators.entropy_balance > params.entropy_gap)
                & (indicators.two_day_change > 0)] = 1
        raw.loc[ready & (indicators.entropy_balance < -params.entropy_gap)
                & (indicators.two_day_change < 0)] = -1
        confirmed = raw.where(raw.eq(raw.shift(1)), 0)
        df = pd.DataFrame({"signal": 0, "size": 0.01}, index=data.index)
        position, age, exposure = 0, 0, 0.01
        highest, lowest, stop, short_stop = -np.inf, np.inf, -np.inf, np.inf
        for i in range(len(data)):
            close = float(data.close.iloc[i])
            atr = float(indicators.atr.iloc[i])
            if position:
                # This close belongs to a session held after the delayed entry.
                age += 1
                if np.isfinite(close) and np.isfinite(atr) and atr > 0:
                    highest = max(highest, close)
                    lowest = min(lowest, close)
                    stop = max(stop, highest - params.atr_multiple * atr)
                    short_stop = min(short_stop, lowest + params.atr_multiple * atr)
                    # The required high-water exit applies to both directions.
                    # Shorts also have a mirrored adverse-move trailing exit.
                    stopped = close <= stop or (position < 0 and close >= short_stop)
                else:
                    stopped = True
                if stopped or age >= 2:
                    position = 0
            elif confirmed.iloc[i] != 0 and np.isfinite(close):
                position = int(confirmed.iloc[i])
                age = 0
                exposure = float(indicators.exposure.iloc[i])
                highest, lowest, stop, short_stop = -np.inf, np.inf, -np.inf, np.inf
            df.iloc[i, 0] = position
            df.iloc[i, 1] = exposure
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = OrdinalParams()
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100., 101., 102., 103., 102., 101., 100., 99., 98.])
    data = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close,
                         "volume": 1000.}, index=index)
    derived = pd.DataFrame({
        "entropy_balance": [0.3, 0.3, 0.3, 0.3, -0.3, -0.3, -0.3, -0.3, -0.3],
        "two_day_change": [1., 1., 1., 1., -1., -1., -1., -1., -1.],
        "exposure": 0.5, "atr": 1.}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result.signal.tolist() == [0, 0, 1, 1, 0, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    cold = strategy.indicators(data, params)
    assert strategy.generate_signals(data, cold, None, params).data.signal.eq(0).all()
    assert strategy.warmup_bars(params) >= params.entropy_window + 2


if __name__ == "__main__":
    _self_check()
