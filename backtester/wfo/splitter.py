from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass(slots=True)
class Window:
    train_data: pd.DataFrame
    test_data: pd.DataFrame


class WalkForwardSplitter:
    """Rolling-origin train/test splitter."""

    def split(
        self,
        data: pd.DataFrame,
        train_bars: int,
        test_bars: int,
        step_bars: int,
    ) -> List[Window]:
        n = len(data)
        if n < train_bars + test_bars:
            raise ValueError(
                f"data too short for WFO: have {n} bars, "
                f"need at least train_bars + test_bars = {train_bars + test_bars}"
            )

        windows: List[Window] = []
        start = 0
        while start + train_bars + test_bars <= n:
            train_end = start + train_bars
            test_end = min(train_end + test_bars, n)
            windows.append(Window(
                train_data=data.iloc[start:train_end].copy(),
                test_data=data.iloc[train_end:test_end].copy(),
            ))
            start += step_bars
        return windows
