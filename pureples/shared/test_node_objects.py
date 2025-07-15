# tests/test_node_objects.py

import pytest
from pureples.es_hyperneat.es_hyperneat import QuadPoint, OctPoint  # 定义于 es_hyperneat.py :contentReference[oaicite:0]{index=0}

def test_quadpoint_initialization():
    """
    测试 QuadPoint(0.1, -0.2, 0.5, 3) 的属性赋值与默认值。
    """
    qp = QuadPoint(0.1, -0.2, 0.5, 3)
    # 入参保存
    assert qp.x == 0.1
    assert qp.y == -0.2
    assert qp.width == 0.5
    assert qp.lvl == 3
    # 默认权重
    assert qp.w == 0.0
    # cs 列表长度及初始内容
    assert isinstance(qp.cs, list)
    assert len(qp.cs) == 4
    assert all(child is None for child in qp.cs)

def test_octpoint_initialization():
    """
    测试 OctPoint(-0.3, 0.4, 0.7, 0.25, 2) 的属性赋值与默认值。
    """
    op = OctPoint(-0.3, 0.4, 0.7, 0.25, 2)
    # 入参保存
    assert op.x == -0.3
    assert op.y == 0.4
    assert op.z == 0.7
    assert op.width == 0.25
    assert op.lvl == 2
    # 默认权重
    assert op.w == 0.0
    # cs 列表长度及初始内容
    assert isinstance(op.cs, list)
    assert len(op.cs) == 8
    assert all(child is None for child in op.cs)
