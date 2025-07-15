# tests/test_division_initialization_3d.py

import pytest
import numpy as np
from pureples.shared.coordinate import Coordinate
from pureples.es_hyperneat.es_hyperneat import ESNetwork, OctPoint  # OctPoint 定义 :contentReference[oaicite:0]{index=0}
from pureples.shared.substrate import Substrate

class DummyCPPN:
    """
    桩对象：activate 返回恒定 [1.0, 1.0]，保证 c.w 非零且可预测。
    """
    def activate(self, vec):
        return [1.0, 1.0]

# 满足 ESNetwork.__init__ 所需的最小参数集
BASE_PARAMS = {
    "initial_depth":      1,
    "max_depth":          1,
    "variance_threshold": 0.0,
    "band_threshold":     0.0,
    "iteration_level":    1,
    "division_threshold": 0.0,
    "max_weight":         1.0,
    "activation":        "sigmoid"
}

@pytest.fixture
def es_network():
    cppn = DummyCPPN()
    # substrate 在 division_initialization_3d 中不被使用，只需合法构造
    substrate = Substrate([(0.0, 0.0)], [(0.0, 0.0)])
    return ESNetwork(substrate, cppn, BASE_PARAMS)

def test_child_coordinates_correctness(es_network):
    """
    测试 division_initialization_3d 生成的 root 是 OctPoint，宽度和层级正确，
    子节点坐标对应 offsets = (±0.5,±0.5,±0.5) 组合。
    """
    root = es_network.division_initialization_3d(Coordinate(0.0, 0.0, 0.0), True)
    # 根节点属性
    assert isinstance(root, OctPoint)
    assert root.width == pytest.approx(1.0)
    assert root.lvl == 1
    # 8 个子节点
    assert len(root.cs) == 8

    # hs = width/2 = 0.5, offsets 顺序见源代码 :contentReference[oaicite:1]{index=1}
    hs = 0.5
    expected_offsets = [
        (-hs, -hs, -hs), (-hs, -hs,  hs),
        (-hs,  hs, -hs), (-hs,  hs,  hs),
        ( hs, -hs, -hs), ( hs, -hs,  hs),
        ( hs,  hs, -hs), ( hs,  hs,  hs),
    ]
    for child, (dx, dy, dz) in zip(root.cs, expected_offsets):
        assert child is not None
        assert child.x == pytest.approx(dx)
        assert child.y == pytest.approx(dy)
        assert child.z == pytest.approx(dz)

def test_variance_consistency(es_network):
    """
    验证 ESNetwork.variance(p) 与 numpy.var( ESNetwork.get_weights(p) ) 一致。
    """
    root = es_network.division_initialization_3d(Coordinate(0.0, 0.0, 0.0), True)
    # 手动收集权重列表
    weights = ESNetwork.get_weights(root)  # get_weights 实现 :contentReference[oaicite:2]{index=2}
    var_manual = np.var(weights)
    var_es = es_network.variance(root)     # variance 实现 :contentReference[oaicite:3]{index=3}
    assert abs(var_es - var_manual) < 1e-9

def test_structure_integrity(es_network):
    """
    验证初始深度=1 时不会继续分裂（所有子节点 lvl == 2），
    并且所有 c.w 均等于 query_cppn 返回的映射值。
    """
    root = es_network.division_initialization_3d(Coordinate(0.0, 0.0, 0.0), True)
    # cs 长度
    assert len(root.cs) == 8
    # 每个子节点的 lvl == root.lvl + 1
    assert all(child.lvl == root.lvl + 1 for child in root.cs)

    # 根据 DummyCPPN.activate 返回 raw = 1.0，
    # query_cppn 映射：(1.0 - 0.2) / 0.8 * max_weight = 1.0 :contentReference[oaicite:4]{index=4}
    expected_w = (1.0 - 0.2) / 0.8 * BASE_PARAMS["max_weight"]
    for child in root.cs:
        assert child.w == pytest.approx(expected_w)
