from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    rank_window: int = 63
    history_window: int = 252
    score_window: int = 126
    atr_window: int = 14
    hold_days: int = 4
    min_analogs: int = 8
    tail_percentile: float = 0.80
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1791271707"

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(max(params.atr_window + 1, params.rank_window + params.hold_days + 2) + params.history_window + params.score_window)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        p = params
        if min(p.rank_window, p.history_window, p.score_window, p.atr_window, p.min_analogs) < 2 or not 3 <= p.hold_days <= 5 or not 0.5 < p.tail_percentile < 1 or p.stop_atr <= 0:
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["settlement"] = (data["close"] - data["low"]) / out["range"]
        out["relative_range"] = out["range"] / data["close"].where(data["close"] > 0)
        out["settlement_rank"] = out["settlement"].rolling(p.rank_window).rank(pct=True)
        out["range_rank"] = out["relative_range"].rolling(p.rank_window).rank(pct=True)
        valid = out[["settlement_rank", "range_rank"]].notna().all(axis=1)
        out["shape_state"] = ((out["settlement_rank"] > 0.5).astype(int) + 2 * (out["range_rank"] > 0.5).astype(int)).where(valid)
        out["motif"] = 4 * out["shape_state"].shift(1) + out["shape_state"]
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([out["range"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        maturity = p.hold_days + 1
        out["mature_motif"] = out["motif"].shift(maturity)
        # At bar t, this outcome belongs to the motif at t-hold_days-1.
        # Its entry was the following open; its exit is today's known open.
        out["mature_return"] = (data["open"] / data["open"].shift(p.hold_days) - 1) / (out["atr"].shift(maturity) / data["close"].shift(maturity)).replace(0, np.nan)
        out["mature_return"] = out["mature_return"].replace([np.inf, -np.inf], np.nan)
        out["forecast"] = np.nan
        out["analog_count"] = 0.0
        # ponytail: sixteen coarse motifs; finer bins need substantially more history.
        for motif in range(16):
            match = out["mature_motif"].eq(motif) & out["mature_return"].notna()
            returns = out["mature_return"].where(match, 0.0)
            count = match.astype(float).rolling(p.history_window).sum()
            mean = returns.rolling(p.history_window).sum() / count.replace(0, np.nan)
            variance = (returns.pow(2).rolling(p.history_window).sum() / count.replace(0, np.nan) - mean.pow(2)).clip(lower=0)
            stderr = np.sqrt(variance / (count - 1).where(count > 1))
            out[f"count_{motif}"] = count
            out[f"mean_{motif}"] = mean
            out[f"stderr_{motif}"] = stderr
            score = (mean / stderr.where(stderr > 0)).where(count >= p.min_analogs)
            out[f"score_{motif}"] = score.replace([np.inf, -np.inf], np.nan)
            current = out["motif"].eq(motif)
            out.loc[current, "forecast"] = out.loc[current, f"score_{motif}"]
            out.loc[current, "analog_count"] = count.loc[current]
        out["forecast_rank"] = out["forecast"].rolling(p.score_window, min_periods=max(2, p.score_window // 2)).rank(pct=True)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        p = params
        ready = indicators["forecast_rank"].notna() & indicators["atr"].gt(0) & indicators["analog_count"].ge(p.min_analogs)
        candidate = np.select([
            ready & indicators["forecast"].gt(0) & indicators["forecast_rank"].ge(p.tail_percentile),
            ready & indicators["forecast"].lt(0) & indicators["forecast_rank"].le(1 - p.tail_percentile)
        ], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side, entry_bar = 0, -1
        entry_price, entry_atr = np.nan, np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                invalid = not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(closes[i])
                # Freeze ATR from the entry decision and price from the fill open.
                # Long stop: entry-k*ATR. Short stop: entry+k*ATR.
                stopped = side * (closes[i] - entry_price) < -p.stop_atr * entry_atr
                expired = i - entry_bar + 1 >= p.hold_days
                if invalid or stopped or expired:
                    side = 0
                raw[i] = side
                continue
            if candidate[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(candidate[i])
                entry_bar = i + 1
                entry_price = np.nan
                entry_atr = atr[i]
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[1], "close"] = 97.0
    data.loc[index[4], "close"] = 103.0
    ind = pd.DataFrame({"forecast": np.nan, "forecast_rank": np.nan, "atr": 1.0, "analog_count": 20.0}, index=index)
    ind.loc[index[[0, 6]], ["forecast", "forecast_rank"]] = [1.0, 1.0]
    ind.loc[index[3], ["forecast", "forecast_rank"]] = [-1.0, 0.0]
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, ShapeParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
