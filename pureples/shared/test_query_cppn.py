# tests/test_query_cppn.py

import pytest
from pureples.hyperneat.hyperneat import query_cppn  # 来自 hyperneat.py 中的函数定义 :contentReference[oaicite:0]{index=0}
from pureples.shared.coordinate import Coordinate  # 来自 coordinate.py 中的类定义 :contentReference[oaicite:1]{index=1}

class StubCPPN:
    """
    桩对象：记录 activate 调用时收到的输入向量，并返回 [vec[0], 0.5] 以便于测试。
    """
    def __init__(self):
        self.received_vec = None

    def activate(self, vec):
        # 记录传入的向量
        self.received_vec = vec
        # 返回可预见的输出：第一个元素直接回传，第二个元素固定为 0.5
        return [vec[0], 0.5]

def test_query_cppn_feeds_full_to_cppn_input():
    """
    测试 query_cppn 必须把 Coordinate.to_cppn_input 的 7 维结果完整传递给 CPPN.activate。
    """
    stub = StubCPPN()
    # 源坐标 (1,0,0)，目标坐标 (0,1,0)
    src = Coordinate(1.0, 0.0, 0.0)
    dst = Coordinate(0.0, 1.0, 0.0)
    # 执行 query_cppn，设置 max_weight=1.0，disable LEO
    w = query_cppn(src, dst, True, stub, max_weight=1.0, enable_leo=False)

    # 1. 查看传入 CPPN.activate 的向量长度应为 7
    assert stub.received_vec is not None
    assert len(stub.received_vec) == 7

    # 2. to_cppn_input 输出应为 [x_src, y_src, z_src, x_dst, y_dst, z_dst, bias]
    expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    assert stub.received_vec == expected

    # 3. stub 返回 raw weight = vec[0] = 1.0，经过 dead-zone 和线性映射后：
    #    (1.0 - 0.2) / 0.8 * max_weight = 1.0
    assert pytest.approx(w, rel=1e-6) == 1.0
