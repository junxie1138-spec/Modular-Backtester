from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class WindowPanel:
    """One walk-forward window over a multi-symbol panel."""
    window_idx: int
    train_data: dict[str, pd.DataFrame]
    train_aux: dict[str, pd.DataFrame]
    test_data: dict[str, pd.DataFrame]
    test_aux: dict[str, pd.DataFrame]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(slots=True)
class MultiSymbolWFOSplitter:
    """Walk-forward splitter for multi-symbol panels.

    Slices ALL symbols (and aux data) together by date. Windows advance by step_bars.
    Train slice: [i .. i + train_bars). Test slice: [i + train_bars .. i + train_bars + test_bars).
    Iteration stops when there are insufficient bars for the next test slice.
    """
    train_bars: int
    test_bars: int
    step_bars: int

    def split(
        self,
        *,
        data: dict[str, pd.DataFrame],
        aux_data: dict[str, pd.DataFrame],
    ) -> Iterator[WindowPanel]:
        if not data:
            return
        # Use the first symbol's index as the canonical timeline.
        # All symbols must share the same index (validated upstream).
        first_sym = next(iter(data))
        index = data[first_sym].index
        n = len(index)

        i = 0
        window_idx = 0
        while i + self.train_bars + self.test_bars <= n:
            train_slice = slice(i, i + self.train_bars)
            test_slice = slice(i + self.train_bars, i + self.train_bars + self.test_bars)
            train_data = {sym: df.iloc[train_slice] for sym, df in data.items()}
            train_aux = {sym: df.iloc[train_slice] for sym, df in aux_data.items()}
            test_data = {sym: df.iloc[test_slice] for sym, df in data.items()}
            test_aux = {sym: df.iloc[test_slice] for sym, df in aux_data.items()}

            yield WindowPanel(
                window_idx=window_idx,
                train_data=train_data,
                train_aux=train_aux,
                test_data=test_data,
                test_aux=test_aux,
                train_start=index[i],
                train_end=index[i + self.train_bars - 1],
                test_start=index[i + self.train_bars],
                test_end=index[i + self.train_bars + self.test_bars - 1],
            )
            i += self.step_bars
            window_idx += 1
