# tests/test_a2_config_objects.py
from __future__ import annotations

import os
import sys
import importlib
from functools import lru_cache
from typing import Any, Iterable, Tuple, Optional

import pytest


# ----------------------------
# Import helpers (abs + fallback)
# ----------------------------
MOD_NAME = "pureples.experiments.retina.es_hyperneat_retina"
SUBSTRATE_MOD = "pureples.shared.substrate"


@lru_cache()
def _import_retina_mod():
    try:
        return importlib.import_module(MOD_NAME)
    except (ModuleNotFoundError, ImportError):
        return importlib.import_module(f"..{MOD_NAME}", package=__package__)


@lru_cache()
def _import_substrate_cls():
    try:
        mod = importlib.import_module(SUBSTRATE_MOD)
    except (ModuleNotFoundError, ImportError):
        mod = importlib.import_module(
            f"..{SUBSTRATE_MOD}", package=__package__)
    return getattr(mod, "Substrate")


def _fresh_import_retina_mod(env_use_3d: Optional[str] = None):
    """强制重新导入被测模块，用于验证导入时的环境同步。"""
    if env_use_3d is None:
        os.environ.pop("ES_USE_3D", None)
    else:
        os.environ["ES_USE_3D"] = env_use_3d

    sys.modules.pop(MOD_NAME, None)
    try:
        return importlib.import_module(MOD_NAME)
    except (ModuleNotFoundError, ImportError):
        return importlib.import_module(f"..{MOD_NAME}", package=__package__)


# ----------------------------
# Geometry normalization helper (fixed)
# ----------------------------
def _norm_sorted_xyz(seq: Iterable[Any], ndigits: int = 9) -> list[Tuple[float, float, float]]:
    """
    把可能是 Coordinate 或 (x,y)/(x,y,z) 的序列统一为
    排序后的 (x, y, z) 浮点三元组列表。为避免浮点微抖动，先 round 再排序。
    注意：不使用 pytest.approx，以免在排序时触发比较器异常。
    """
    out: list[Tuple[float, float, float]] = []
    for item in seq:
        if hasattr(item, "x") and hasattr(item, "y"):
            x = float(getattr(item, "x"))
            y = float(getattr(item, "y"))
            z = float(getattr(item, "z", 0.0))
        else:
            t = tuple(item)
            if len(t) == 2:
                x, y = map(float, t)
                z = 0.0
            elif len(t) == 3:
                x, y, z = map(float, t)
            else:
                raise AssertionError(f"Unsupported coord arity: {t!r}")
        out.append((round(x, ndigits), round(y, ndigits), round(z, ndigits)))
    out.sort()
    return out


# ----------------------------
# Tests
# ----------------------------
def test_T1_cfg_path_resolves():
    mod = _import_retina_mod()
    cfg_path = getattr(mod, "CFG_PATH", None)
    assert cfg_path is not None, "CFG_PATH should be exported"
    assert os.path.exists(str(cfg_path)), f"CFG file not found: {cfg_path}"


def test_T2_es_params_keys_and_types():
    mod = _import_retina_mod()
    es = getattr(mod, "ES_PARAMS")
    assert isinstance(es.get("enable_leo"), bool)
    assert isinstance(es.get("leo_threshold"), float)
    loc = es.get("locality_seed")
    assert (loc is None) or isinstance(loc, str)
    assert isinstance(es.get("use_3d"), bool)


def test_T3_config_syncs_to_es_params():
    mod = _import_retina_mod()
    es = getattr(mod, "ES_PARAMS")
    cfg = getattr(mod, "CONFIG")
    assert cfg is not None and hasattr(cfg, "genome_config")
    assert isinstance(es.get("enable_leo"), bool)
    assert isinstance(es.get("leo_threshold"), float)


def test_T4_substrate_exports():
    mod = _import_retina_mod()
    Substrate = _import_substrate_cls()
    sub = getattr(mod, "SUBSTRATE")
    assert isinstance(sub, Substrate)

    # 以几何等价为准（归一化到 (x,y,z) 后比较）
    assert _norm_sorted_xyz(sub.input_coordinates) == _norm_sorted_xyz(
        getattr(mod, "INPUT_COORDS")
    )
    assert _norm_sorted_xyz(sub.output_coordinates) == _norm_sorted_xyz(
        getattr(mod, "OUTPUT_COORDS")
    )


def test_T5_apply_use_3d_true_updates_config_and_es_params():
    mod = _import_retina_mod()
    mod.apply_use_3d(True)
    es = getattr(mod, "ES_PARAMS")
    assert es["use_3d"] is True
    gc = mod.CONFIG.genome_config
    assert isinstance(gc.num_inputs, int) and gc.num_inputs > 0
    assert isinstance(gc.input_keys, (list, tuple)) and len(
        gc.input_keys) == gc.num_inputs


def test_T6_apply_use_3d_false_updates_config_and_es_params():
    mod = _import_retina_mod()
    mod.apply_use_3d(False)
    es = getattr(mod, "ES_PARAMS")
    assert es["use_3d"] is False
    gc = mod.CONFIG.genome_config
    assert isinstance(gc.num_inputs, int) and gc.num_inputs > 0
    assert isinstance(gc.input_keys, (list, tuple)) and len(
        gc.input_keys) == gc.num_inputs


def test_T7_apply_use_3d_roundtrip_consistency():
    mod = _import_retina_mod()
    mod.apply_use_3d(True)
    gc1 = (mod.CONFIG.genome_config.num_inputs,
           tuple(mod.CONFIG.genome_config.input_keys))
    mod.apply_use_3d(False)
    gc2 = (mod.CONFIG.genome_config.num_inputs,
           tuple(mod.CONFIG.genome_config.input_keys))
    mod.apply_use_3d(True)
    gc3 = (mod.CONFIG.genome_config.num_inputs,
           tuple(mod.CONFIG.genome_config.input_keys))
    assert gc1 == gc3
    assert gc1 != gc2


def test_T8_import_syncs_from_env_true(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ES_USE_3D", "1")
    mod = _fresh_import_retina_mod(env_use_3d="1")
    es = getattr(mod, "ES_PARAMS")
    assert es["use_3d"] is True
    gc = mod.CONFIG.genome_config
    assert isinstance(gc.input_keys, (list, tuple)) and len(
        gc.input_keys) == gc.num_inputs


def test_T9_import_syncs_from_env_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ES_USE_3D", "0")
    mod = _fresh_import_retina_mod(env_use_3d="0")
    es = getattr(mod, "ES_PARAMS")
    assert es["use_3d"] is False
    gc = mod.CONFIG.genome_config
    assert isinstance(gc.input_keys, (list, tuple)) and len(
        gc.input_keys) == gc.num_inputs


def test_A1_blackbox_acceptance_I_only():
    """I 维黑盒整体验收：仅校验导出的对象/属性 Schema 与不变式。"""
    mod = _import_retina_mod()

    cfg_path = getattr(mod, "CFG_PATH", None)
    assert cfg_path is not None and os.path.exists(str(cfg_path))

    gc = mod.CONFIG.genome_config
    assert isinstance(gc.num_inputs, int) and gc.num_inputs > 0
    assert isinstance(gc.input_keys, (list, tuple)) and len(
        gc.input_keys) == gc.num_inputs

    es = getattr(mod, "ES_PARAMS")
    assert isinstance(es.get("enable_leo"), bool)
    assert isinstance(es.get("leo_threshold"), float)
    loc = es.get("locality_seed")
    assert (loc is None) or isinstance(loc, str)

    Substrate = _import_substrate_cls()
    sub = getattr(mod, "SUBSTRATE")
    assert isinstance(sub, Substrate)
    assert _norm_sorted_xyz(
        sub.input_coordinates) == _norm_sorted_xyz(mod.INPUT_COORDS)
    assert _norm_sorted_xyz(
        sub.output_coordinates) == _norm_sorted_xyz(mod.OUTPUT_COORDS)
