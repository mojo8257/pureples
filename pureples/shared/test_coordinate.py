# tests/test_coordinate.py

import pytest
from pureples.shared.coordinate import Coordinate

def test_default_z_axis():
    """
    如果只传入 x 和 y，z 应当默认为 0.0
    """
    c = Coordinate(0.5, -0.5)
    assert c.x == 0.5
    assert c.y == -0.5
    assert c.z == 0.0

def test_normalize_clamp():
    """
    坐标值超出 [-1,1] 时，应被截断到边界
    """
    c = Coordinate(2.0, -3.0, 5.0)
    assert c.x == 1.0
    assert c.y == -1.0
    assert c.z == 1.0

def test_to_cppn_input_3d():
    """
    在三维情况下（显式传入 z），to_cppn_input 应返回 [x1, y1, z1, x2, y2, z2, 1.0]
    """
    src = Coordinate(0.0, 0.0, 0.0)
    dst = Coordinate(1.0, 1.0, 1.0)
    vec = src.to_cppn_input(dst, outgoing=True)
    assert vec == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

def test_to_cppn_input_2d():
    """
    在二维情况下（z 默认为 0），to_cppn_input 也要补齐 z 并返回 7 维向量
    """
    src = Coordinate(0.0, 0.0)
    dst = Coordinate(1.0, 1.0)
    vec = src.to_cppn_input(dst, outgoing=True)
    assert vec == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]

def test_equality_and_type():
    """
    相同数值的两个 Coordinate 应视为相等，不同类型返回 False
    """
    c1 = Coordinate(0.0, 0.0)
    c2 = Coordinate(0.0, 0.0)
    assert c1 == c2
    assert not (c1 == (0.0, 0.0, 0.0))  # 不同类型

def test_to_tuple():
    """
    to_tuple 方法应当返回 (x, y, z) 的三元组
    """
    c = Coordinate(0.1, 0.2)
    assert c.to_tuple() == (0.1, 0.2, 0.0)
