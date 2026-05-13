from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd


class DataLoader(ABC):
    def __init__(self, root: Path):
        self.root = Path(root)

    @abstractmethod
    def load(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError
