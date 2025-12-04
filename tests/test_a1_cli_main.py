# /workspaces/pureples/tests/test_a1_cli_main.py
# -*- coding: utf-8 -*-

import os
import sys
import importlib
from functools import lru_cache

import pytest


@lru_cache()
def _import_retina_mod():
    """
    绝对导入优先，失败再按你的规则用相对定位退路。
    """
    try:
        return importlib.import_module(
            "pureples.experiments.retina.es_hyperneat_retina"
        )
    except (ModuleNotFoundError, ImportError):
        # 注意：tests 顶层 __package__ 可能为空，这里显式给 package="pureples"
        return importlib.import_module(
            "..experiments.retina.es_hyperneat_retina", package="pureples"
        )


def _detect_use3d_flag(mod) -> bool:
    """
    使用“真实被 apply_use_3d 切换的布尔开关”判断 2D/3D，而不是去看坐标 z。
    探测顺序：
      1) 模块级布尔位：USE_3D/use_3d/IS_3D/is_3d
      2) ES_PARAMS 中包含 '3d' 且为 bool 的字段（优先 use_3d/enable_3d/fully_3d）
      3) 兜底（仅 main 场景）：环境变量 ES_USE_3D
    找不到则抛出断言，避免误判。
    """
    # 1) 模块级布尔位
    for name in ("USE_3D", "use_3d", "IS_3D", "is_3d"):
        v = getattr(mod, name, None)
        if isinstance(v, bool):
            return v

    # 2) ES_PARAMS
    params = getattr(mod, "ES_PARAMS", None)
    if params is not None:
        # 对象 or dict 都处理
        def _iter_items(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield k, v
            else:
                for k in dir(obj):
                    # 过滤私有/dunder
                    if k.startswith("_"):
                        continue
                    yield k, getattr(obj, k)

        cands = []
        for k, v in _iter_items(params):
            if "3d" in k.lower() and isinstance(v, bool):
                cands.append((k, v))

        if len(cands) == 1:
            return cands[0][1]

        for key in ("use_3d", "enable_3d", "fully_3d"):
            if hasattr(params, key) and isinstance(getattr(params, key), bool):
                return getattr(params, key)

    # 3) 仅 main 的兜底（apply_use_3d 不应写 env，但 main 会）
    env = os.environ.get("ES_USE_3D")
    if env in ("1", "true", "True"):
        return True
    if env in ("0", "false", "False"):
        return False

    raise AssertionError(
        "无法从模块或 ES_PARAMS 中可靠探测 use_3d 开关；请确认项目暴露的 3D 标志。"
    )


# ---------------------- 测试用例 ----------------------


def test_T1_apply_true_sets_3d_idempotent():
    mod = _import_retina_mod()

    # 预设 2D
    mod.apply_use_3d(False)
    assert _detect_use3d_flag(mod) is False

    # 切到 3D
    mod.apply_use_3d(True)
    assert _detect_use3d_flag(mod) is True

    # 幂等：再次 True 仍然 True
    mod.apply_use_3d(True)
    assert _detect_use3d_flag(mod) is True


def test_T2_apply_false_sets_2d_idempotent():
    mod = _import_retina_mod()

    # 预设 3D
    mod.apply_use_3d(True)
    assert _detect_use3d_flag(mod) is True

    # 切回 2D
    mod.apply_use_3d(False)
    assert _detect_use3d_flag(mod) is False

    # 幂等：再次 False 仍然 False
    mod.apply_use_3d(False)
    assert _detect_use3d_flag(mod) is False


def test_T3_apply_reversible_roundtrip():
    mod = _import_retina_mod()
    mod.apply_use_3d(True)
    assert _detect_use3d_flag(mod) is True
    mod.apply_use_3d(False)
    assert _detect_use3d_flag(mod) is False
    mod.apply_use_3d(True)
    assert _detect_use3d_flag(mod) is True


def test_T4_apply_does_not_touch_env(monkeypatch: pytest.MonkeyPatch):
    """
    apply_use_3d 只应修改内部开关，不改环境变量（职责边界）。
    """
    mod = _import_retina_mod()
    monkeypatch.delenv("ES_USE_3D", raising=False)

    mod.apply_use_3d(True)
    assert _detect_use3d_flag(mod) is True
    assert "ES_USE_3D" not in os.environ

    mod.apply_use_3d(False)
    assert _detect_use3d_flag(mod) is False
    assert "ES_USE_3D" not in os.environ


def test_T5_main_dry_run_switches_by_real_apply(monkeypatch: pytest.MonkeyPatch):
    """
    --dry-run + --use-3d 应走 main，并通过真实的 apply_use_3d 切到 3D，然后退出。
    """
    mod = _import_retina_mod()

    # baseline 设为 2D
    mod.apply_use_3d(False)
    assert _detect_use3d_flag(mod) is False

    # 捕获退出
    def _exit_stub(code=None):
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", _exit_stub, raising=False)

    argv = ["prog", "--dry-run", "--use-3d", "--generations", "4"]
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    with pytest.raises(SystemExit) as ei:
        mod.main()
    assert (ei.value.code or 0) == 0

    # 验证：真实 apply_use_3d 已执行
    assert _detect_use3d_flag(mod) is True


def test_T6_main_writes_env_apply_does_not(monkeypatch: pytest.MonkeyPatch):
    """
    main 负责写入环境变量 ES_USE_3D；apply_use_3d 不动环境变量。
    """
    mod = _import_retina_mod()
    monkeypatch.delenv("ES_USE_3D", raising=False)

    # 截获退出
    def _exit_stub(code=None):
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", _exit_stub, raising=False)
    monkeypatch.setattr(
        sys, "argv", ["prog", "--dry-run", "--use-3d", "--generations", "2"], raising=False
    )

    with pytest.raises(SystemExit):
        mod.main()

    # main 写 env
    assert os.environ.get("ES_USE_3D") == "1"
    assert _detect_use3d_flag(mod) is True

    # apply_use_3d(False) 不应覆盖 env（职责边界），但内部开关应变为 False
    mod.apply_use_3d(False)
    assert os.environ.get("ES_USE_3D") == "1"
    assert _detect_use3d_flag(mod) is False
