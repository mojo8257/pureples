import pytest
from pureples.shared.coordinate import Coordinate
from pureples.shared.substrate import Substrate
from pureples.es_hyperneat.es_hyperneat import ESNetwork, OctPoint, Connection3D  # OctPoint 定义 :contentReference[oaicite:0]{index=0}, Connection3D 定义 :contentReference[oaicite:1]{index=1}

class DummyCppn:
    """
    按序返回预置的 raw outputs，以便 query_cppn 映射后产生可预测的邻居权重。
    """
    def __init__(self, raws):
        self.raws = raws
        self.idx = 0
    def activate(self, vec):
        # query_cppn 只关心 outputs[0]
        raw = self.raws[self.idx]
        self.idx += 1
        return [raw, 0.0]

# 最小合法参数集：只覆盖 pruning_extraction_3d 所需的字段
BASE_PARAMS = {
    "initial_depth":      1,
    "max_depth":          1,
    "variance_threshold": 0.03,   # 保证 variance(c)=0.0 < 阈值 ⇒ 执行剪枝分支
    "band_threshold":     0.2,    # 用于测试阈值判定
    "iteration_level":    1,
    "division_threshold": 0.0,
    "max_weight":         1.0,    # 方便 mapping：w = (raw-0.2)/0.8 * 1.0
    "activation":        "sigmoid",
    "use_3d":             True
}

def make_network(raws):
    """根据给定 raws 列表构造 ESNetwork 实例。"""
    cppn = DummyCppn(raws)
    # substrate 仅需合法构造，不会在本测试中使用
    substrate = Substrate([(0.0, 0.0)], [(0.0, 1.0)])  # :contentReference[oaicite:2]{index=2}
    return ESNetwork(substrate, cppn, BASE_PARAMS)

def make_point():
    """构造一个宽度 0.5、位于原点的单一子节点容器 p，以及唯一子节点 c。"""
    p = OctPoint(0.0, 0.0, 0.0, 0.5, lvl=1)
    c = OctPoint(0.0, 0.0, 0.0,    0.1, lvl=2)
    c.w = 1.0  # 直接指定为映射后的权重 1.0
    p.cs = [c]  # 只保留一个子节点以便迭代
    return p, c

def test_connection_created_when_band_exceeds_threshold():
    """
    测试项 A & B(a)：当沿 x 方向的最小差异 band_x = 0.3 > threshold(0.2) 时，
    pruning_extraction_3d 应向 connections 中添加一条 Connection3D。
    """
    # 为 6 个方向提供 raw outputs，使得映射后邻居权重 = [1.3,1.3,0.9,0.9,0.8,0.8]
    # raw = 0.8 * target + 0.2
    target_ws = [1.3, 1.3, 0.9, 0.9, 0.8, 0.8]
    raws = [0.8*t + 0.2 for t in target_ws]
    es = make_network(raws)
    p, c = make_point()
    coord = Coordinate(0.0, 0.0, 0.0)

    es.pruning_extraction_3d(coord, p, outgoing=True)  # 实现见 prune 3D 分支 :contentReference[oaicite:3]{index=3}

    # 应只产生一条连接
    assert len(es.connections) == 1

def test_no_connection_when_all_bands_below_threshold():
    """
    测试项 B(b)：当所有轴的最小差异都 = 0.05 < threshold(0.2) 时，不应产生连接。
    """
    # 统一 raw 使得映射后邻居权重 = 1.05 ⇒ diff = 0.05
    raw = 0.8*1.05 + 0.2
    raws = [raw] * 6
    es = make_network(raws)
    p, c = make_point()
    coord = Coordinate(0.0, 0.0, 0.0)

    es.pruning_extraction_3d(coord, p, outgoing=True)

    assert len(es.connections) == 0

def test_no_connection_when_band_equals_threshold():
    """
    测试项 B(c)：当最小差异恰好等于 threshold(0.2) 时，也不应产生连接。
    """
    # 选映射后邻居权重 = 1.2 ⇒ diff = 0.2
    raw = 0.8*1.2 + 0.2
    raws = [raw] * 6
    es = make_network(raws)
    p, c = make_point()
    coord = Coordinate(0.0, 0.0, 0.0)

    es.pruning_extraction_3d(coord, p, outgoing=True)

    assert len(es.connections) == 0

def test_connection3d_recording_and_hash_behavior():
    """
    测试项 C：验证 Connection3D 的记录准确性以及 __hash__/__eq__。
    """
    # 重用场景 (a)，确保仅产生一条连接
    target_ws = [1.3, 1.3, 0.9, 0.9, 0.8, 0.8]
    raws = [0.8*t + 0.2 for t in target_ws]
    es = make_network(raws)
    p, c = make_point()
    coord = Coordinate(0.0, 0.0, 0.0)

    es.pruning_extraction_3d(coord, p, outgoing=True)
    assert len(es.connections) == 1

    con = next(iter(es.connections))
    # 类型与属性校验
    assert isinstance(con, Connection3D)
    assert con.src.to_tuple() == (0.0, 0.0, 0.0)
    assert con.dst.to_tuple() == c.to_tuple()
    assert con.weight == pytest.approx(c.w)

    # 再次添加等价连接，集合大小不变
    es.connections.add(Connection3D(con.src, con.dst, con.weight))
    assert len(es.connections) == 1