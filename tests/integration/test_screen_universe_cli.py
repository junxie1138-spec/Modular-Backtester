import subprocess
import sys
from pathlib import Path


def test_screen_universe_smoke(tmp_path):
    seed = tmp_path / "seed.txt"
    seed.write_text("TSLA\nNVDA\nAMD\n")
    out = tmp_path / "universe_candidates.yaml"
    result = subprocess.run(
        [
            sys.executable, "scripts/screen_universe.py",
            "--candidates", str(seed),
            "--start", "2023-01-03", "--end", "2024-12-31",
            "--out", str(out), "--top", "10",
            "--data-root", "data/raw",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    import yaml
    doc = yaml.safe_load(out.read_text())
    assert "universe" in doc
