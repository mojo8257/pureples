# tests/test_b23_forward_inference.py
# -*- coding: utf-8 -*-
"""
B2.3 表型前向推理（retina_eval_single 的前向部分）集成测试
覆盖：T1~T6
导入遵循规则：绝对导入优先；仅捕获 ImportError/ModuleNotFoundError 做相对退路；
类型仅在 TYPE_CHECKING；惰性导入；统一导入助手使用 @lru_cache。
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple
import importlib

import pytest

if TYPE_CHECKING:  # 类型与运行时分离
    from pureples.es_hyperneat.es_hyperneat import ESNetwork as ESNetworkType  # noqa: F401
    from pureples.shared.substrate import Substrate as SubstrateType  # noqa: F401
    from pureples.shared.coordinate import Coordinate as CoordinateType  # noqa: F401


# ------------------------------------------------------------------
# 导入助手：绝对导入优先；失败用相对退路（仅捕获 ImportError/ModuleNotFoundError）
# ------------------------------------------------------------------
@lru_cache()
def _import_modules():
    try:
        es_mod = importlib.import_module("pureples.es_hyperneat.es_hyperneat")
    except (ImportError, ModuleNotFoundError):
        es_mod = importlib.import_module(
            "..pureples.es_hyperneat.es_hyperneat", package=__package__)

    try:
        shared_coord = importlib.import_module("pureples.shared.coordinate")
    except (ImportError, ModuleNotFoundError):
        shared_coord = importlib.import_module(
            "..pureples.shared.coordinate", package=__package__)

    try:
        shared_substrate = importlib.import_module("pureples.shared.substrate")
    except (ImportError, ModuleNotFoundError):
        shared_substrate = importlib.import_module(
            "..pureples.shared.substrate", package=__package__)

    try:
        neat = importlib.import_module("neat")
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError("需要 neat-python，请先安装：pip install neat-python") from e

    ESNetwork = es_mod.ESNetwork
    Connection3D = es_mod.Connection3D
    Coordinate = shared_coord.Coordinate
    Substrate = shared_substrate.Substrate

    return ESNetwork, Connection3D, Coordinate, Substrate, neat


# ------------------------------------------------------------------
# CPPN 桩：用于 ESNetwork 构网阶段的 query_cppn 调用
# ------------------------------------------------------------------
class CPPNStub:
    def __init__(self, w_raw: float = 1.0, leo: float | None = None,
                 fn: Callable[[Sequence[float]], Sequence[float]] | None = None):
        self._w_raw = w_raw
        self._leo = leo
        self._fn = fn

    def activate(self, vec: Sequence[float]) -> List[float]:
        if self._fn is not None:
            return list(self._fn(vec))
        outs = [float(self._w_raw)]
        if self._leo is not None:
            outs.append(float(self._leo))
        return outs


# ------------------------------------------------------------------
# Phenotype 桩：用于 T1/T2，直接验证输入映射与解包契约
# ------------------------------------------------------------------
class PhenStub:
    def __init__(self, num_inputs: int, out_pair: Tuple[float, float] = (1.23, 4.56)):
        self.input_nodes = list(range(num_inputs))
        self._out_pair = out_pair
        self.last_vec: List[float] | None = None

    def activate(self, vec: Sequence[float]) -> Tuple[float, float]:
        assert len(vec) == len(self.input_nodes), "input length mismatch"
        self.last_vec = list(vec)
        return self._out_pair


# ------------------------------------------------------------------
# 工具：输入映射（等价于 retina_eval_single L138）
# ------------------------------------------------------------------
def to_inputs(pattern: Sequence[int | bool]) -> List[float]:
    return [3.0 if bool(p) else -3.0 for p in pattern]


# ------------------------------------------------------------------
# 工具：构建真实表型（B2.1+B2.2 合成）
# - 极简 substrate：N 输入，2 输出
# - monkeypatch ESNetwork.es_hyperneat → 输入全连两输出，保证能前向
# ------------------------------------------------------------------
def build_real_phenotype(num_inputs: int = 6,
                         use_3d: bool = False,
                         enable_leo: bool = False,
                         leo_threshold: float = 0.0,
                         monkeypatch: pytest.MonkeyPatch | None = None):
    ESNetwork, Connection3D, _Coordinate, Substrate, neat = _import_modules()

    # 1) Substrate：N 输入（x 轴均匀），2 输出（上侧两点）
    step = 1.0 / max(1, num_inputs)
    inputs_xy = [(-0.5 + (i + 0.5) * step, -0.5) for i in range(num_inputs)]
    outputs_xy = [(-0.25, 0.5), (0.25, 0.5)]
    sub = Substrate(input_coordinates=inputs_xy,
                    output_coordinates=outputs_xy, hidden_coordinates=[], res=10.0)

    # 2) ES 参数（宽松阈值，保证表达）
    params = dict(
        initial_depth=1, max_depth=1, variance_threshold=0.0, band_threshold=0.0,
        iteration_level=1, division_threshold=0.0, max_weight=5.0, activation="sigmoid",
        enable_leo=enable_leo, leo_threshold=leo_threshold, use_3d=use_3d,
    )

    # 3) CPPN：常量正权；若开 LEO 则返回 leo=1.0 通过门控
    cppn = CPPNStub(w_raw=1.0, leo=(1.0 if enable_leo else None))

    # 4) 伪装 es_hyperneat：输入全连两输出
    def _es_min(self: "ESNetworkType"):
        # Substrate 已经提供了 Coordinate 对象列表，直接使用即可
        src_nodes = list(self.substrate.input_coordinates)
        out_nodes = list(self.substrate.output_coordinates)
        conns = {Connection3D(src=s, dst=o, weight=1.0)
                 for s in src_nodes for o in out_nodes}
        hidden = set()
        return hidden, conns

    if monkeypatch is not None:
        monkeypatch.setattr(ESNetwork, "es_hyperneat", _es_min, raising=False)

    # 5) 表型
    phen = ESNetwork(sub, cppn, params).create_phenotype_network()
    assert hasattr(phen, "activate"), "phenotype must provide .activate(vec)"
    return phen


# ------------------------------------------------------------------
# 烟雾收集测试：确保本文件至少收集到 1 条测试（排查“no tests ran”）
# ------------------------------------------------------------------
def test__collector_smoke():
    assert True


# =========================================================
# T1（B4-I，B5-I）：输入映射 全真/全假/混合
# =========================================================
@pytest.mark.parametrize(
    "pattern, expect",
    [
        ([1, 1, 1, 1], [3.0, 3.0, 3.0, 3.0]),
        ([0, 0, 0, 0], [-3.0, -3.0, -3.0, -3.0]),
        ([1, 0, 1, 0], [3.0, -3.0, 3.0, -3.0]),
    ],
)
def test_T1_input_mapping(pattern, expect):
    phen = PhenStub(num_inputs=len(pattern))
    inp = to_inputs(pattern)
    out_left, out_right = phen.activate(inp)
    assert phen.last_vec == expect
    assert isinstance(out_left, (int, float)) and isinstance(
        out_right, (int, float))


# =========================================================
# T2（B1-I，B10-I）：输出为 2 通道且可解包
# =========================================================
def test_T2_two_channel_unpack():
    phen = PhenStub(num_inputs=3, out_pair=(1.23, 4.56))
    out_left, out_right = phen.activate(to_inputs([1, 0, 1]))
    assert out_left == pytest.approx(1.23)
    assert out_right == pytest.approx(4.56)


# =========================================================
# T3（B7-I，B8-I）：长度一致性（2D/3D）
# =========================================================
@pytest.mark.parametrize("use_3d", [False, True])
def test_T3_length_consistency_2d_3d(use_3d, monkeypatch):
    n = 6
    phen = build_real_phenotype(
        num_inputs=n, use_3d=use_3d, monkeypatch=monkeypatch)
    assert hasattr(phen, "input_nodes") and hasattr(phen, "output_nodes")
    pattern = [1 if i % 2 == 0 else 0 for i in range(len(phen.input_nodes))]
    vec = to_inputs(pattern)
    out_left, out_right = phen.activate(vec)
    assert isinstance(out_left, (int, float)) and isinstance(
        out_right, (int, float))


# =========================================================
# T4（B2-D，B3-D）：形态驱动黑盒（LEO / 阈值 / locality-seed / 2D↔3D）
# =========================================================
@pytest.mark.parametrize(
    "use_3d, enable_leo, leo_threshold",
    [
        (False, False, 0.0),   # Baseline
        (False, True, 0.5),    # LEO 开
        (False, True, 0.5),    # LEO + xaxis（B2.3 对前向无直接影响，此处与上等价）
        (True,  False, 0.0),   # 3D
    ],
)
def test_T4_shapes_across_phenotypes(use_3d, enable_leo, leo_threshold, monkeypatch):
    n = 8
    phen = build_real_phenotype(
        num_inputs=n,
        use_3d=use_3d,
        enable_leo=enable_leo,
        leo_threshold=leo_threshold,
        monkeypatch=monkeypatch,
    )
    pattern = [1 if i % 3 else 0 for i in range(len(phen.input_nodes))]
    vec = to_inputs(pattern)
    out = phen.activate(vec)
    assert isinstance(out, (list, tuple)) and len(out) == 2
    _l, _r = out  # 解包不应抛错


# =========================================================
# T5（B9-E）：异常契约（长度不一致时报错）
# =========================================================
def test_T5_length_mismatch_raises(monkeypatch):
    phen = build_real_phenotype(
        num_inputs=5, use_3d=False, monkeypatch=monkeypatch)
    bad_vec = to_inputs([1] * (len(phen.input_nodes) - 1))
    with pytest.raises((AssertionError, ValueError, IndexError, RuntimeError)):
        phen.activate(bad_vec)
    bad_vec2 = to_inputs([1] * (len(phen.input_nodes) + 1))
    with pytest.raises((AssertionError, ValueError, IndexError, RuntimeError)):
        phen.activate(bad_vec2)


# =========================================================
# T6（B6-I）：边界图案（空图案或最小像素）
# =========================================================
def test_T6_boundary_pattern_empty_or_minimal(monkeypatch):
    phen = build_real_phenotype(
        num_inputs=1, use_3d=False, monkeypatch=monkeypatch)
    if len(phen.input_nodes) > 0:
        with pytest.raises((AssertionError, ValueError, IndexError, RuntimeError)):
            phen.activate([])  # 空图案
    else:
        out = phen.activate([])
        assert isinstance(out, (list, tuple)) and len(out) == 2
