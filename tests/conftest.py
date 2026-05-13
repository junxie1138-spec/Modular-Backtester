from __future__ import annotations

import pytest

from tests.fixtures.synthetic import make_ohlcv


@pytest.fixture
def ohlcv_small():
    return make_ohlcv(n=60, seed=1)


@pytest.fixture
def ohlcv_medium():
    return make_ohlcv(n=750, seed=1)
