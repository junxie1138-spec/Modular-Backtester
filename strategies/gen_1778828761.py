from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownDepthParams:
    lookback_high: int = 60
    entry_drawdown_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    holding_bars: int = 18
    spike_zscore: float = 2.5
    spike_window: int = 60
    refractory_bars: int = 5
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[DrawdownDepthParams]):
    strategy_id = "gen_1778828761"

    @classmethod
    def params_type(cls) -> type[DrawdownDepthParams]:
        return DrawdownDepthParams

    @classmethod
    def warmup_bars(cls, params: DrawdownDepthParams) -> int:
        return int(max(params.lookback_high, params.spike_window) + 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownDepthParams) -> pd.DataFrame:
        close = data["close"].astype(float)

        rolling_peak = close.rolling(params.lookback_high, min_periods=params.lookback_high).max()
        drawdown = (close - rolling_peak) / rolling_peak

        daily_ret = close.pct_change()
        ret_mean = daily_ret.rolling(params.spike_window, min_periods=params.spike_window).mean()
        ret_std = daily_ret.rolling(params.spike_window, min_periods=params.spike_window).std()
        safe_std = ret_std.replace(0.0, np.nan)
        ret_zscore = (daily_ret - ret_mean) / safe_std

        spike_flag = (ret_zscore < -params.spike_zscore).fillna(False).astype(int)
        refractory_window = max(int(params.refractory_bars), 1)
        spike_recent = (
            spike_flag.rolling(refractory_window, min_periods=1).sum().fillna(0.0) > 0
        ).astype(int)

        ind = pd.DataFrame(index=data.index)
        ind["drawdown"] = drawdown
        ind["ret_zscore"] = ret_zscore
        ind["spike_flag"] = spike_flag
        ind["spike_recent"] = spike_recent
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownDepthParams,
    ) -> SignalFrame:
        n = len(data)
        drawdown = indicators["drawdown"].fillna(0.0).to_numpy()
        spike_recent = indicators["spike_recent"].fillna(0).astype(int).to_numpy()

        dd_abs = np.clip(-drawdown, 0.0, None)
        in_drawdown = dd_abs >= params.entry_drawdown_pct
        not_refractory = spike_recent == 0
        eligible = in_drawdown & not_refractory

        span = max(params.max_drawdown_pct - params.entry_drawdown_pct, 1e-9)
        floor_size = float(max(params.min_size, 0.01))
        ceil_size = float(max(params.max_size, floor_size))

        in_position = np.zeros(n, dtype=np.int64)
        pos_size = np.full(n, floor_size, dtype=np.float64)

        hold_left = 0
        current_size = floor_size
        hold_target = max(int(params.holding_bars), 1)

        for i in range(n):
            if hold_left > 0:
                in_position[i] = 1
                pos_size[i] = current_size
                hold_left -= 1
            elif eligible[i]:
                scaled = (dd_abs[i] - params.entry_drawdown_pct) / span
                if scaled < 0.0:
                    scaled = 0.0
                elif scaled > 1.0:
                    scaled = 1.0
                current_size = floor_size + scaled * (ceil_size - floor_size)
                in_position[i] = 1
                pos_size[i] = current_size
                hold_left = hold_target - 1
            else:
                current_size = floor_size

        signal = pd.Series(in_position, index=data.index, dtype="int64")
        signal = signal.shift(1).fillna(0).astype(int)

        size = pd.Series(pos_size, index=data.index, dtype="float64")
        size = size.shift(1).fillna(floor_size).astype(float).clip(lower=0.01)

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
