from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 20
    smoothing: int = 5
    low_entropy: float = 0.90
    high_entropy: float = 0.97
    entry_strength: float = 0.06
    sizing_scale: float = 0.30
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791269104"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if not (params.pattern_window >= 6 and params.smoothing >= 1
                and 0 <= params.low_entropy < params.high_entropy <= 1
                and 0 < params.entry_strength < 1
                and params.sizing_scale > 0
                and 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid ordinal-complexity parameters")
        return int(params.pattern_window + params.smoothing + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        out["return_lag1"] = out["return"].shift(1)
        out["return_lag2"] = out["return"].shift(2)
        a, b, c = out["return_lag2"], out["return_lag1"], out["return"]
        complete = a.notna() & b.notna() & c.notna()
        distinct = complete & a.ne(b) & a.ne(c) & b.ne(c)
        code = (a.gt(b).astype(int) + 2 * a.gt(c).astype(int)
                + 4 * b.gt(c).astype(int))
        out["pattern"] = code.where(distinct)
        window = params.pattern_window
        out["pattern_count"] = distinct.astype(float).rolling(window).sum()
        ready = (complete.astype(float).rolling(window).sum().eq(window)
                 & out["pattern_count"].ge(0.8 * window))
        denominator = out["pattern_count"].replace(0, np.nan)
        columns = []
        for pattern in (0, 1, 3, 4, 6, 7):
            name = f"p{pattern}"
            out[name] = out["pattern"].eq(pattern).astype(float).rolling(
                window
            ).sum().div(denominator).where(ready)
            columns.append(name)
        probabilities = out[columns]
        out["entropy_raw"] = -(probabilities * np.log(
            probabilities.where(probabilities.gt(0), 1.0)
        )).sum(axis=1).div(np.log(6.0)).where(ready)
        out["pressure_raw"] = out["p0"] + out["p1"] - out["p6"] - out["p7"]
        out["entropy"] = out["entropy_raw"].rolling(params.smoothing).mean()
        out["pressure"] = out["pressure_raw"].rolling(params.smoothing).mean()
        switches = pd.Series(np.where(
            out["entropy"].le(params.low_entropy), 1.0,
            np.where(out["entropy"].ge(params.high_entropy), -1.0, np.nan)
        ), index=data.index)
        out["regime"] = switches.ffill().fillna(1.0)
        out["score"] = out["pressure"] * out["regime"]
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        score = indicators["score"].replace([np.inf, -np.inf], np.nan)
        entry_condition = score.ge(params.entry_strength) & score.notna()
        # Exiting is exactly the reversal of the active entry condition.
        df["signal"] = entry_condition.astype(int)
        conviction = score.abs().div(params.sizing_scale).clip(0.0, 1.0)
        df["size"] = (params.min_size + (
            params.max_size - params.min_size
        ) * conviction).fillna(params.min_size)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    strategy = GeneratedStrategy()
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 240)))
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000000.0
    }, index=pd.date_range("2020-01-01", periods=len(close), freq="B"))
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, None, params).data
    expected = indicators["score"].ge(params.entry_strength).astype(int)
    pd.testing.assert_series_equal(
        signals["signal"], expected.shift(1).fillna(0).astype(int),
        check_names=False
    )
    pd.testing.assert_frame_equal(
        indicators.iloc[:160], strategy.indicators(data.iloc[:160], params)
    )
    assert signals["signal"].isin([0, 1]).all()
    assert signals["size"].between(params.min_size, params.max_size).all()
    assert signals["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_indicators = strategy.indicators(flat, params)
    assert strategy.generate_signals(
        flat, flat_indicators, None, params
    ).data["signal"].eq(0).all()
