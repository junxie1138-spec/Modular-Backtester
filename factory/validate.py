from __future__ import annotations

import ast
import logging
from typing import Iterable

import yaml

log = logging.getLogger(__name__)

ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset({
    "pandas", "numpy", "dataclasses", "__future__", "typing",
    "backtester", "backtester.core", "backtester.core.types",
    "backtester.strategies", "backtester.strategies.base",
})

REQUIRED_METHODS: tuple[str, ...] = (
    "params_type", "warmup_bars", "indicators", "generate_signals",
)


class StaticValidationError(ValueError):
    """Tier 1 static check failure."""


class FunctionalValidationError(ValueError):
    """Tier 2 functional check failure (implemented in Task 6)."""


def _import_root(node: ast.AST) -> Iterable[str]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name.split(".")[0]
    elif isinstance(node, ast.ImportFrom):
        if node.module is None:
            return
        yield node.module.split(".")[0]


def _check_imports(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for root in _import_root(node):
                if root not in ALLOWED_IMPORT_ROOTS and not any(
                    a == root or a.startswith(root + ".") for a in ALLOWED_IMPORT_ROOTS
                ):
                    raise StaticValidationError(
                        f"forbidden import root: {root!r} (allowed: {sorted(ALLOWED_IMPORT_ROOTS)})"
                    )


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise StaticValidationError(f"class {name!r} not found")


def _class_methods(cls: ast.ClassDef) -> set[str]:
    return {
        n.name for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_assigns(cls: ast.ClassDef) -> dict[str, ast.AST]:
    """Return a {name -> value-node} map of class-body name = value assignments."""
    out: dict[str, ast.AST] = {}
    for n in cls.body:
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = n.value
    return out


def _has_dataclass_slots_true(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for dec in node.decorator_list:
            # @dataclass(slots=True)
            if isinstance(dec, ast.Call):
                func = dec.func
                if (isinstance(func, ast.Name) and func.id == "dataclass") or (
                    isinstance(func, ast.Attribute) and func.attr == "dataclass"
                ):
                    for kw in dec.keywords:
                        if kw.arg == "slots" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            return True
    return False


def validate_static(
    *,
    strategy_id: str,
    strategy_src: str,
    config_src: str,
    allow_short: bool,
) -> None:
    """Tier 1 static contract checks. Raises StaticValidationError on first failure."""
    # 1. Parses.
    try:
        tree = ast.parse(strategy_src)
    except SyntaxError as exc:
        raise StaticValidationError(f"strategy file does not parse: {exc}") from exc

    # 5. Import whitelist (run early — cheap and catches the worst offenders).
    _check_imports(tree)

    # 2. GeneratedStrategy class present.
    cls = _find_class(tree, "GeneratedStrategy")

    # 3. Required methods.
    methods = _class_methods(cls)
    missing = [m for m in REQUIRED_METHODS if m not in methods]
    if missing:
        raise StaticValidationError(
            f"GeneratedStrategy missing required methods: {missing}"
        )

    # 4. strategy_id attribute present and matches injected id.
    assigns = _class_assigns(cls)
    if "strategy_id" not in assigns:
        raise StaticValidationError("GeneratedStrategy missing strategy_id attribute")
    val = assigns["strategy_id"]
    if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
        raise StaticValidationError("strategy_id must be a string literal")
    if val.value != strategy_id:
        raise StaticValidationError(
            f"strategy_id mismatch: file declares {val.value!r}, injected {strategy_id!r}"
        )

    # Factory-specific: forbid v0.4.0 multi-symbol opt-in attributes.
    for forbidden in ("uses_multi_symbol", "uses_per_bar"):
        if forbidden in assigns:
            v = assigns[forbidden]
            if isinstance(v, ast.Constant) and v.value is True:
                raise StaticValidationError(
                    f"{forbidden} = True is forbidden (factory targets v0.3.0-style strategies only)"
                )

    # 6. Shift present (cheap proxy for the mandatory one-bar shift).
    if ".shift(1)" not in strategy_src:
        raise StaticValidationError(
            "strategy source does not contain '.shift(1)' (the mandatory one-bar signal shift)"
        )

    # 7. @dataclass(slots=True) params class present.
    if not _has_dataclass_slots_true(tree):
        raise StaticValidationError(
            "no @dataclass(slots=True) found (params class is required)"
        )

    # 8. Config sanity.
    try:
        cfg = yaml.safe_load(config_src)
    except yaml.YAMLError as exc:
        raise StaticValidationError(f"config_file does not parse: {exc}") from exc
    if not isinstance(cfg, dict):
        raise StaticValidationError("config_file root must be a mapping")
    if cfg.get("strategy") != strategy_id:
        raise StaticValidationError(
            f"config strategy={cfg.get('strategy')!r} does not match strategy_id={strategy_id!r}"
        )
    wfo = cfg.get("wfo") or {}
    if not wfo.get("enabled", False):
        raise StaticValidationError("config wfo.enabled must be true")
    exec_block = cfg.get("execution") or {}
    if bool(exec_block.get("allow_short", False)) != bool(allow_short):
        raise StaticValidationError(
            f"config execution.allow_short={exec_block.get('allow_short')} "
            f"does not match strategy allow_short={allow_short}"
        )


import importlib.util
import sys
import uuid
from pathlib import Path

import pandas as pd


def _load_strategy_module(src: str, tmp_dir: Path) -> object:
    """Load a strategy source file as a one-off module without touching the
    registered strategies. Returns the imported module object.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mod_name = f"_factory_validate_{uuid.uuid4().hex}"
    path = tmp_dir / f"{mod_name}.py"
    path.write_text(src, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise FunctionalValidationError(f"could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module
    except FunctionalValidationError:
        raise
    except Exception as exc:
        raise FunctionalValidationError(
            f"strategy failed to import: {type(exc).__name__}: {exc}"
        ) from exc


def validate_functional(
    *,
    strategy_id: str,
    strategy_src: str,
    allow_short: bool,
    tmp_dir: Path,
) -> None:
    """Tier 2 functional smoke test.

    Imports the strategy in isolation (no registry pollution), instantiates it,
    runs indicators() + generate_signals() against a 200-bar synthetic OHLCV
    frame, and asserts the SignalFrame contract.
    """
    # Import dependencies lazily so unit tests of Tier 1 don't need pandas
    # to load this module.
    from backtester.core.types import SignalFrame, StrategyContext
    from factory.synth_ohlcv import make_synthetic_ohlcv

    module = _load_strategy_module(strategy_src, tmp_dir)

    cls = getattr(module, "GeneratedStrategy", None)
    if cls is None:
        raise FunctionalValidationError("imported module has no GeneratedStrategy")

    # Instantiate. params_type() should be a dataclass with all-defaulted fields.
    try:
        params_cls = cls.params_type()
        params = params_cls()
        strategy = cls()
    except Exception as exc:
        raise FunctionalValidationError(
            f"strategy/params instantiation failed: {type(exc).__name__}: {exc}"
        ) from exc

    # warmup_bars sanity.
    try:
        warmup = int(strategy.warmup_bars(params))
    except Exception as exc:
        raise FunctionalValidationError(f"warmup_bars raised: {exc}") from exc
    if warmup < 0 or warmup > 1000:
        raise FunctionalValidationError(f"warmup_bars out of sane range: {warmup}")

    data = make_synthetic_ohlcv(n_bars=max(200, warmup + 50))

    # Indicators.
    try:
        ind = strategy.indicators(data, params)
    except Exception as exc:
        raise FunctionalValidationError(
            f"indicators() raised: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(ind, pd.DataFrame):
        raise FunctionalValidationError(
            f"indicators() returned {type(ind).__name__}, expected DataFrame"
        )
    if not ind.index.equals(data.index):
        raise FunctionalValidationError("indicators index does not match data index")

    # Signals.
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=warmup)
    try:
        sf = strategy.generate_signals(data, ind, ctx, params)
    except Exception as exc:
        raise FunctionalValidationError(
            f"generate_signals() raised: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(sf, SignalFrame):
        raise FunctionalValidationError(
            f"generate_signals returned {type(sf).__name__}, expected SignalFrame"
        )

    df = sf.data
    if not isinstance(df, pd.DataFrame):
        raise FunctionalValidationError("SignalFrame.data is not a DataFrame")
    if "signal" not in df.columns:
        raise FunctionalValidationError("SignalFrame missing 'signal' column")
    if not df.index.equals(data.index):
        raise FunctionalValidationError("SignalFrame index does not match data index")

    sig = df["signal"]
    # Integer dtype required.
    if not pd.api.types.is_integer_dtype(sig):
        raise FunctionalValidationError(
            f"signal column dtype must be integer, got {sig.dtype}"
        )

    unique = set(int(x) for x in sig.dropna().unique())
    allowed = {-1, 0, 1} if allow_short else {0, 1}
    extra = unique - allowed
    if extra:
        raise FunctionalValidationError(
            f"signal values outside allowed set {sorted(allowed)}: found {sorted(extra)}"
        )

    # size column required and positive.
    if sf.size_column is None or sf.size_column not in df.columns:
        raise FunctionalValidationError("SignalFrame missing 'size' column")
    size = df[sf.size_column]
    if (size <= 0).any():
        raise FunctionalValidationError("size column contains non-positive values")

    # First-bar zero (cheap one-bar-shift sanity).
    if int(sig.iloc[0]) != 0:
        raise FunctionalValidationError(
            f"first signal must be 0 (one-bar-shift sanity); got {int(sig.iloc[0])}"
        )
