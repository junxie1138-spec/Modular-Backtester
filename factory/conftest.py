import sys
from pathlib import Path

# Make `from factory.<module>` and `from backtester.<module>` both resolve
# when pytest is invoked from any cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
