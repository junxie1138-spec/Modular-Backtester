from pathlib import Path
import csv


SECTOR_MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "sector_map.csv"

REQUIRED_SYMBOLS = {
    "TSLA", "NVDA", "AMD", "COIN", "GOOGL", "MSTR", "XPEV", "NIO",
    "PLTR", "SMCI", "SHOP", "W", "META", "NFLX", "SPY", "^VIX",
}


def test_sector_map_csv_parses():
    assert SECTOR_MAP_PATH.exists(), f"missing {SECTOR_MAP_PATH}"
    with SECTOR_MAP_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["symbol", "sector"], (
            f"unexpected header: {reader.fieldnames}"
        )
        rows = list(reader)
    assert len(rows) >= len(REQUIRED_SYMBOLS)


def test_every_required_symbol_has_sector_entry():
    with SECTOR_MAP_PATH.open(newline="", encoding="utf-8") as f:
        rows = {row["symbol"]: row["sector"] for row in csv.DictReader(f)}
    missing = REQUIRED_SYMBOLS - set(rows)
    assert not missing, f"sector_map.csv missing: {missing}"
    for sym, sector in rows.items():
        assert sector, f"{sym} has empty sector"
