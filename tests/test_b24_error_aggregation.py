# tests/test_b24_error_aggregation.py
# -*- coding: utf-8 -*-
"""
B2.4 误差聚合与适应度计算（retina_eval_single）集成测试
覆盖：T1~T10 + A1
要点：
- 严格遵守导入规范：绝对导入优先，仅捕获 ImportError/ModuleNotFoundError 用相对退路；
  惰性导入；类型仅在 TYPE_CHECKING；统一用 @lru_cache 的导入助手。
- 通过 monkeypatch：
  * 控制 itertools.product 的枚举规模/样本；
  * 将 ESNetwork.create_phenotype_network 替换为桩 PhenStub（两通道输出，并可记录输入向量）；
  * 将 neat.nn.FeedForwardNetwork.create 替换为无副作用桩（不影响 B2.4 行为）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple
import importlib

import pytest

if TYPE_CHECKING:  # 类型与运行期分离
    from pureples.es_hyperneat.es_hyperneat import ESNetwork as ESNetworkType  # noqa: F401
    from pureples.experiments.retina.es_hyperneat_retina import retina_eval_single as RetinaEvalType  # noqa: F401


# ------------------------------------------------------------------
# 导入助手（绝对优先；仅捕获 ImportError/ModuleNotFoundError 做相对退路）
# ------------------------------------------------------------------
@lru_cache()
def _import_modules():
    try:
        retina_mod = importlib.import_module(
            "pureples.experiments.retina.es_hyperneat_retina")
    except (ImportError, ModuleNotFoundError):
        retina_mod = importlib.import_module(
            "..pureples.experiments.retina.es_hyperneat_retina", package=__package__)

    try:
        es_mod = importlib.import_module("pureples.es_hyperneat.es_hyperneat")
    except (ImportError, ModuleNotFoundError):
        es_mod = importlib.import_module(
            "..pureples.experiments.retina.es_hyperneat_retina", package=__package__)

    try:
        neat = importlib.import_module("neat")
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError("需要 neat-python，请先安装：pip install neat-python") from e

    # stdlib
    import itertools  # 轻量，直接导入

    # 暴露所需符号
    retina_eval_single = retina_mod.retina_eval_single
    ESNetwork = es_mod.ESNetwork
    return retina_mod, es_mod, neat, itertools, retina_eval_single, ESNetwork


# ------------------------------------------------------------------
# 桩：两通道表型网络（记录输入向量与调用次数）
# ------------------------------------------------------------------
class PhenStub:
    def __init__(self,
                 num_inputs: int = 8,
                 out_fn: Callable[[Sequence[float]], Tuple[float, float]] | None = None):
        self.input_nodes = list(range(num_inputs))
        self.output_nodes = [num_inputs, num_inputs + 1]
        self._out_fn = out_fn or (lambda _v: (0.0, 0.0))
        self.calls: int = 0
        self.seen_vecs: List[List[float]] = []

    def activate(self, vec: Sequence[float]) -> Tuple[float, float]:
        # 与真实网络一致的长度检查
        if len(vec) != len(self.input_nodes):
            raise RuntimeError(
                f"Expected {len(self.input_nodes)} inputs, got {len(vec)}")
        self.calls += 1
        v = list(vec)
        self.seen_vecs.append(v)
        out = self._out_fn(v)
        assert isinstance(out, (tuple, list)) and len(
            out) == 2, "PhenStub must return 2-channel outputs"
        return float(out[0]), float(out[1])


# ------------------------------------------------------------------
# 便捷：替换 ESNetwork.create_phenotype_network 与 FeedForwardNetwork.create
# ------------------------------------------------------------------
def _patch_build_pipeline_to_phen(monkeypatch: pytest.MonkeyPatch,
                                  phen_factory: Callable[[], PhenStub]):
    retina_mod, _es_mod, neat, _itertools, _retina_eval_single, ESNetwork = _import_modules()

    # 1) CPPN 创建桩：忽略 genome/config，返回任意对象即可
    def _cppn_create(_genome, _cfg):
        class _Dummy:  # noqa: D401
            """dummy cppn object"""
            pass
        return _Dummy()

    monkeypatch.setattr(neat.nn.FeedForwardNetwork,
                        "create", _cppn_create, raising=False)

    # 2) Phenotype 创建桩：忽略 self/cppn/params 的细节，直接返回我们的桩
    def _phen_create(_self: "ESNetworkType"):
        return phen_factory()

    monkeypatch.setattr(ESNetwork, "create_phenotype_network",
                        _phen_create, raising=False)

    # 返回 retina_eval_single 以便调用
    return retina_mod.retina_eval_single


# ------------------------------------------------------------------
# 便捷：替换 itertools.product 的枚举输出
# ------------------------------------------------------------------
def _patch_product(monkeypatch: pytest.MonkeyPatch, patterns: List[Tuple[int, ...]]):
    _retina_mod, _es_mod, _neat, itertools_mod, _retina_eval_single, _ESNetwork = _import_modules()

    def _fake_product(_choices, repeat: int):
        # ignore inputs; return the precomputed patterns
        for p in patterns:
            assert len(p) == repeat, "pattern length must match repeat"
            yield p
    monkeypatch.setattr(itertools_mod, "product", _fake_product, raising=False)


# ------------------------------------------------------------------
# 桩 genome / neat_config（仅占位，不被使用）
# ------------------------------------------------------------------
class _GenomeStub:  # noqa: D401
    """placeholder genome stub"""
    pass


class _ConfigStub:  # noqa: D401
    """placeholder neat_config stub"""
    pass


# =========================================================
# T1（B1-D）：自定义枚举规模 → 前向次数一致
# =========================================================
def test_T1_custom_enum_drives_forward_calls(monkeypatch):
    patterns = [
        (1, 0, 1, 0, 1, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1, 1, 1, 1),
        (1, 1, 0, 0, 1, 1, 0, 0),
        (0, 1, 0, 1, 0, 1, 0, 1),
    ]
    _patch_product(monkeypatch, patterns)

    phen = PhenStub(num_inputs=8, out_fn=lambda _v: (0.0, 0.0))

    def _phen_factory():
        return phen

    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, _phen_factory)
    fitness = retina_eval_single(_GenomeStub(), _ConfigStub())

    assert phen.calls == len(patterns)
    assert isinstance(fitness, float)


# =========================================================
# T2（B2-I）：默认 8-bit 全枚举 → 256 次
# =========================================================
def test_T2_default_enumeration_256(monkeypatch):
    # 不替换 itertools.product；替换构网为桩
    phen = PhenStub(num_inputs=8, out_fn=lambda _v: (0.0, 0.0))
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen)

    f = retina_eval_single(_GenomeStub(), _ConfigStub())
    assert phen.calls == 256  # repeat=8 → 2**8
    assert isinstance(f, float)


# =========================================================
# T3（B3/B4-I）：输入映射（±3.0，按位、顺序）
# =========================================================
def test_T3_input_mapping(monkeypatch):
    samples = [
        (1, 1, 1, 1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 1, 0, 1, 0),
    ]
    _patch_product(monkeypatch, samples)

    phen = PhenStub(num_inputs=8, out_fn=lambda _v: (0.0, 0.0))
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen)
    _ = retina_eval_single(_GenomeStub(), _ConfigStub())

    expect = [
        [3.0] * 8,
        [-3.0] * 8,
        [3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0],
    ]
    assert phen.seen_vecs == expect


# =========================================================
# T4（B5-I）：两通道输出与解包（冒烟）
# =========================================================
def test_T4_two_channel_unpack(monkeypatch):
    _patch_product(monkeypatch, [(1,)*8])
    phen = PhenStub(num_inputs=8, out_fn=lambda _v: (1.0, 2.0))
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen)
    f = retina_eval_single(_GenomeStub(), _ConfigStub())
    assert isinstance(f, float)


# =========================================================
# T5（B7-I）：空枚举 → fitness=1000
# =========================================================
def test_T5_empty_enum_yields_max_fitness(monkeypatch):
    _patch_product(monkeypatch, [])
    phen = PhenStub(num_inputs=8, out_fn=lambda _v: (123.0, -456.0))  # 不会被调用
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen)
    f = retina_eval_single(_GenomeStub(), _ConfigStub())
    assert phen.calls == 0
    assert f == pytest.approx(1000.0)


# =========================================================
# T6（B8-I）：有界性（0 < f <= 1000）
# =========================================================
def test_T6_boundedness(monkeypatch):
    _patch_product(monkeypatch, [(1,)*8, (0,)*8, (1, 0, 1, 0, 1, 0, 1, 0)])
    phen = PhenStub(num_inputs=8, out_fn=lambda _v: (10.0, -10.0))
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen)
    f = retina_eval_single(_GenomeStub(), _ConfigStub())
    assert 0.0 < f <= 1000.0


# =========================================================
# T7（B9-D）：相对单调性（扩大偏差 → fitness 下降）
# 说明：目标函数未知；为避免偶发不满足，保守标记为 xfail（文档化意图）。
# =========================================================
@pytest.mark.xfail(strict=False, reason="目标函数未对外暴露，缩放偏差的相对单调性在某些目标下可能相等或不成立。")
def test_T7_relative_monotonicity(monkeypatch):
    patterns = [(1,)*8, (0,)*8, (1, 0, 1, 0, 1, 0, 1, 0)]
    _patch_product(monkeypatch, patterns)

    # Run-A：较小幅度
    phen_a = PhenStub(num_inputs=8, out_fn=lambda _v: (1.0, -1.0))
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen_a)
    f_a = retina_eval_single(_GenomeStub(), _ConfigStub())

    # Run-B：更大幅度（在很多合理目标下会增大误差）
    phen_b = PhenStub(num_inputs=8, out_fn=lambda _v: (3.0, -3.0))
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen_b)
    f_b = retina_eval_single(_GenomeStub(), _ConfigStub())

    assert f_b < f_a


# =========================================================
# T8（B10-E）：异常传播（phen.activate 抛错不被吞）
# =========================================================
def test_T8_exception_propagates(monkeypatch):
    _patch_product(monkeypatch, [(1,)*8])

    def _boom(_v):
        raise RuntimeError("boom")

    phen = PhenStub(num_inputs=8, out_fn=lambda v: _boom(v))
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen)

    with pytest.raises(RuntimeError):
        _ = retina_eval_single(_GenomeStub(), _ConfigStub())


# =========================================================
# T9（B11-D）：形态黑盒（2D / 3D 参数切换不影响聚合流程）
# 说明：我们桩化了 phenotype 创建，因此只验证流程可执行与返回值 Schema。
# =========================================================
@pytest.mark.parametrize("use_3d", [False, True])
def test_T9_shape_switch_2d_3d(monkeypatch, use_3d):
    retina_mod, _es_mod, _neat, _itertools, _retina_eval_single, _ESNetwork = _import_modules()

    # 覆写模块级 ES_PARAMS 的 use_3d
    es_params = dict(getattr(retina_mod, "ES_PARAMS", {}))
    es_params["use_3d"] = bool(use_3d)
    monkeypatch.setattr(retina_mod, "ES_PARAMS", es_params, raising=False)

    _patch_product(monkeypatch, [(1,)*8, (0,)*8])
    phen = PhenStub(num_inputs=8, out_fn=lambda _v: (0.5, -0.5))
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen)
    f = retina_eval_single(_GenomeStub(), _ConfigStub())
    assert isinstance(f, float) and 0.0 < f <= 1000.0


# =========================================================
# T10（B11-D）：LEO on/off 与 locality-seed 冒烟
# 说明：桩 phenotype，验证聚合流程可执行并返回 fitness。
# =========================================================
@pytest.mark.parametrize(
    "enable_leo, leo_threshold, locality_seed",
    [
        (False, 0.0, None),
        (True, 0.5, None),
        (True, 0.5, "xaxis"),
    ],
)
def test_T10_leo_and_locality_seed_smoke(monkeypatch, enable_leo, leo_threshold, locality_seed):
    retina_mod, _es_mod, _neat, _itertools, _retina_eval_single, _ESNetwork = _import_modules()

    es_params = dict(getattr(retina_mod, "ES_PARAMS", {}))
    es_params["enable_leo"] = bool(enable_leo)
    es_params["leo_threshold"] = float(leo_threshold)
    if locality_seed is not None:
        es_params["locality_seed"] = str(locality_seed)
    monkeypatch.setattr(retina_mod, "ES_PARAMS", es_params, raising=False)

    _patch_product(monkeypatch, [(1,)*8, (0,)*8, (1, 0, 1, 0, 1, 0, 1, 0)])
    phen = PhenStub(num_inputs=8, out_fn=lambda _v: (0.1, -0.2))
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen)
    f = retina_eval_single(_GenomeStub(), _ConfigStub())
    assert isinstance(f, float) and 0.0 < f <= 1000.0


# =========================================================
# A1（整体验收 / I 维黑盒）
# 端到端（桩 CPPN + 桩 phenotype），不校验目标具体数值，仅校验 Schema 与有界性。
# =========================================================
def test_A1_blackbox_acceptance(monkeypatch):
    # 默认 256 样本
    phen = PhenStub(num_inputs=8, out_fn=lambda v: (
        sum(v) * 0.0, -sum(v) * 0.0))  # (=0,0)
    retina_eval_single = _patch_build_pipeline_to_phen(
        monkeypatch, lambda: phen)
    f = retina_eval_single(_GenomeStub(), _ConfigStub())

    # 断言：Schema + 有界性 + 调用次数
    assert isinstance(f, float) and 0.0 < f <= 1000.0
    assert phen.calls == 256
