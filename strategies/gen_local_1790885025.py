from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 12
    baseline_window: int = 36
    flow_window: int = 6
    atr_window: int = 14
    entropy_drop: float = 0.12
    ordinal_bias: float = 0.10
    flow_shift: float = 0.15
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790885025"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.pattern_window + params.baseline_window + 3,
            2 * params.flow_window,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        x, y, z = out["return"].shift(2), out["return"].shift(1), out["return"]
        # Equal returns are ordered by their chronological position.
        conditions = [
            (x <= y) & (y <= z),
            (x <= z) & (z < y),
            (y < x) & (x <= z),
            (y <= z) & (z < x),
            (z < x) & (x <= y),
            (z < y) & (y < x),
        ]
        out["pattern"] = np.select(conditions, np.arange(6), default=np.nan)
        out["pattern"] = out["pattern"].where(x.notna() & y.notna() & z.notna())
        recent, baseline = [], []
        for k in range(6):
            occurrence = out["pattern"].eq(k).astype(float).where(
                out["pattern"].notna()
            )
            out[f"occurrence_{k}"] = occurrence
            out[f"recent_{k}"] = occurrence.rolling(params.pattern_window).mean()
            out[f"baseline_{k}"] = occurrence.shift(params.pattern_window).rolling(
                params.baseline_window
            ).mean()
            recent.append(out[f"recent_{k}"])
            baseline.append(out[f"baseline_{k}"])
        p = pd.concat(recent, axis=1)
        q = pd.concat(baseline, axis=1)
        out["entropy"] = -(p * np.log(p.where(p > 0, 1.0))).sum(
            axis=1, skipna=False
        ) / np.log(6.0)
        out["baseline_entropy"] = -(q * np.log(q.where(q > 0, 1.0))).sum(
            axis=1, skipna=False
        ) / np.log(6.0)
        out["entropy_drop"] = out["baseline_entropy"] - out["entropy"]
        out["ordinal_skew"] = out["recent_0"] - out["recent_5"]
        out["close_location"] = (
            (2.0 * data["close"] - data["high"] - data["low"])
            / (data["high"] - data["low"]).replace(0.0, np.nan)
        ).clip(-1.0, 1.0)
        out["signed_volume"] = out["close_location"] * data["volume"]
        out["volume_sum"] = data["volume"].rolling(params.flow_window).sum()
        out["flow"] = out["signed_volume"].rolling(params.flow_window).sum() / (
            out["volume_sum"].where(out["volume_sum"] > 0.0)
        )
        out["flow_change"] = out["flow"] - out["flow"].shift(params.flow_window)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        collapsed = out["entropy_drop"].ge(params.entropy_drop)
        out["changepoint"] = collapsed & ~collapsed.shift(1, fill_value=False)
        bullish = (
            out["changepoint"]
            & out["ordinal_skew"].ge(params.ordinal_bias)
            & out["flow"].gt(0.0)
            & out["flow_change"].ge(params.flow_shift)
        )
        bearish = (
            out["changepoint"]
            & out["ordinal_skew"].le(-params.ordinal_bias)
            & out["flow"].lt(0.0)
            & out["flow_change"].le(-params.flow_shift)
        )
        out["entry"] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        out.loc[out.index[:self.warmup_bars(params) - 1], "entry"] = 0
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        targets = np.zeros(len(data), dtype=int)
        proposals = indicators["entry"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position, pending, held = 0, 0, 0
        fixed_stop = np.nan
        for i in range(len(data)):
            # A decision from yesterday becomes a position at today's open.
            if pending:
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0.0:
                    position = pending
                    fixed_stop = opens[i] - position * params.stop_atr * atr[i - 1]
                    held = 0
                pending = 0
            if position:
                held += 1
                stopped = (
                    not np.isfinite(closes[i])
                    or (position == 1 and closes[i] < fixed_stop)
                    or (position == -1 and closes[i] > fixed_stop)
                )
                if stopped or held >= params.holding_bars:
                    position = 0
                targets[i] = position
            elif proposals[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                pending = int(proposals[i])
                targets[i] = pending
        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 95.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    data.iloc[2, data.columns.get_loc("close")] = 96.0
    data.iloc[5, data.columns.get_loc("close")] = 104.0
    indicators = pd.DataFrame({
        "entry": [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0],
        "atr": 1.0,
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(
        data, indicators, None, Params(holding_bars=3)
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 0, 1, 1, 1, 0]
    assert result["size"].gt(0.0).all()
    index = pd.date_range("2021-01-01", periods=160)
    close = 100.0 + np.arange(160) * 0.1 + np.sin(np.arange(160) * 0.7)
    data = pd.DataFrame({
        "open": close - 0.1, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000.0 + np.arange(160),
    }, index=index)
    params = Params()
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:110], params)
    pd.testing.assert_frame_equal(full.iloc[:110], prefix)
    full_signals = strategy.generate_signals(data, full, None, params).data
    prefix_signals = strategy.generate_signals(data.iloc[:110], prefix, None, params).data
    pd.testing.assert_frame_equal(full_signals.iloc[:110], prefix_signals)


if __name__ == "__main__":
    _self_check()
