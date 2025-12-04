# tests/test_b22_phenotype_network.py
# -*- coding: utf-8 -*-
"""
B2.2 表型网络生成（CPPN→表型网络）集成测试
覆盖：依赖集 Deps-1..5 的代表性行为 + A1 黑盒整体验收
聚焦：LEO、band_threshold、索引装配、node_evals 契约、2D/3D 一致性

备注：
- 通过 monkeypatch 精准替换 ESNetwork.es_hyperneat()，构造最小可控的
  hidden_nodes / connections 集合，从而把测试重点放在
  create_phenotype_network() 的索引装配与 node_evals 构造逻辑上。
- Deps-1（LEO 门控、band_threshold）中，为保持与实现一致，
  仍调用真实的 query_cppn() 来决定连接是否表达（由 CPPN 桩返回）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Any
import pytest

from functools import lru_cache
from importlib import import_module

if TYPE_CHECKING:
    # 仅供类型检查使用；运行时不会实际导入
    from pureples.es_hyperneat.es_hyperneat import ESNetwork as ESNetworkType


@lru_cache(maxsize=1)
def _import_modules() -> tuple[object, type, type, Callable[..., Any], Any]:
    """
    返回 (es_mod, Coordinate, Substrate, query_cppn, neat)

    优先使用包内绝对导入；若导入失败，则基于 __package__ 做相对导入兜底。
    仅捕获 ModuleNotFoundError，避免把运行期异常误判为导入失败。
    使用 import_module 避免 Pylance 对相对导入的静态误报。
    """
    try:
        # —— 首选：包方式（正确的绝对路径）——
        es_mod = import_module("pureples.es_hyperneat.es_hyperneat")
        Coordinate = getattr(import_module(
            "pureples.shared.coordinate"), "Coordinate")
        Substrate = getattr(import_module(
            "pureples.shared.substrate"), "Substrate")
        query_cppn = getattr(import_module(
            "pureples.hyperneat.hyperneat"), "query_cppn")
    except ModuleNotFoundError:
        # —— 退路：源码树内相对导入 ——
        pkg = __package__ or "pureples.hyperneat"  # 兜底包名，便于脚本方式运行
        es_mod = import_module("..es_hyperneat.es_hyperneat", package=pkg)
        Coordinate = getattr(import_module(
            "..shared.coordinate", package=pkg), "Coordinate")
        Substrate = getattr(import_module(
            "..shared.substrate", package=pkg), "Substrate")
        query_cppn = getattr(import_module(
            ".hyperneat", package=pkg), "query_cppn")

    # neat 顶级包：来自子模块安装（-e external/neat-python）或 PyPI
    try:
        neat = import_module("neat")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "未安装 neat（neat-python）。请执行：pip install -e external/neat-python  或  pip install neat-python"
        ) from e

    return es_mod, Coordinate, Substrate, query_cppn, neat


es_mod, Coordinate, Substrate, query_cppn, neat = _import_modules()
Connection3D = es_mod.Connection3D
ESNetwork = es_mod.ESNetwork  # 运行期别名（供 monkeypatch 使用）


# --------------------------
# CPPN 桩（可控返回 w_raw / leo）
# --------------------------
class CPPNStub:
    """
    activate(input_vec[7]) -> [w_raw, leo?]
    - 若 fn 不为 None，优先用 fn 计算；否则返回固定 w_raw/leo。
    """

    def __init__(
        self,
        w_raw: float = 1.0,
        leo: Optional[float] = 1.0,
        fn: Optional[Callable[[List[float]], List[float]]] = None,
    ):
        self._w_raw = w_raw
        self._leo = leo
        self._fn = fn

    def activate(self, vec: List[float]) -> List[float]:
        if self._fn is not None:
            return self._fn(vec)
        outs = [self._w_raw]
        if self._leo is not None:
            outs.append(self._leo)
        return outs


# --------------------------
# 公共夹具：Substrate / 参数
# --------------------------
@pytest.fixture
def substrate_1in_1out():
    # 一个输入 (0,0)，一个输出 (0,0.5) —— 避免与输入坐标重合
    return Substrate(
        input_coordinates=[(0.0, 0.0)],
        output_coordinates=[(0.0, 0.5)],
        hidden_coordinates=[],
        res=10.0,
    )


@pytest.fixture
def substrate_2in_2out():
    return Substrate(
        input_coordinates=[(-0.5, 0.0), (0.5, 0.0)],
        output_coordinates=[(-0.5, 0.5), (0.5, 0.5)],
        hidden_coordinates=[],
        res=10.0,
    )


def make_params(**over):
    """
    ESNetwork 所需参数的稳定基线；可用关键字覆盖。
    """
    base = dict(
        initial_depth=1,
        max_depth=2,
        variance_threshold=0.1,
        band_threshold=0.2,
        iteration_level=1,
        division_threshold=0.1,
        max_weight=5.0,
        activation="sigmoid",
        enable_leo=False,
        leo_threshold=0.0,
        use_3d=False,
    )
    base.update(over)
    return base


# --------------------------
# 工具：解析 node_evals
# --------------------------
def _node_eval_map(net):
    """
    将 net.node_evals 转为 {idx: links[(src_id, w), ...]}
    """
    m = {}
    for idx, _act, _agg, _bias, _resp, links in net.node_evals:
        m[idx] = list(links)
    return m


# =========================================================
# T1（Deps-1 / D+I-1）：LEO 门控—阈值下抑制
# =========================================================
def test_T1_leo_gate_below_threshold(monkeypatch, substrate_1in_1out):
    params = make_params(enable_leo=True, leo_threshold=0.5)
    cppn = CPPNStub(w_raw=0.9, leo=0.4)  # LEO < 阈值 → 抑制

    # es_hyperneat 最小替身：用 query_cppn 决定是否连 input->H
    def es_min(self: "ESNetworkType"):
        src = self.substrate.input_coordinates[0]
        cand = Coordinate(0.5, 0.0)  # 一个假隐藏点
        w = query_cppn(
            src,
            cand,
            True,
            self.cppn,
            self.max_weight,
            enable_leo=self.enable_leo,
            leo_threshold=self.leo_threshold,
        )
        hidden, conns = set(), set()
        if w != 0.0:  # 被 LEO 抑制则不进集合
            hidden.add(cand)
            conns.add(Connection3D(src=src, dst=cand, weight=w))
        return hidden, conns

    monkeypatch.setattr(ESNetwork, "es_hyperneat", es_min, raising=False)
    net = ESNetwork(substrate_1in_1out, cppn,
                    params).create_phenotype_network()

    # 断言：没有任何连接与 node_evals（被门控抑制）
    assert net.node_evals == []


# =========================================================
# T2（Deps-1 / I-1）：LEO 门控—阈值上表达
# =========================================================
def test_T2_leo_gate_above_threshold(monkeypatch, substrate_1in_1out):
    params = make_params(enable_leo=True, leo_threshold=0.5)
    cppn = CPPNStub(w_raw=0.9, leo=0.6)  # LEO >= 阈值 → 表达

    def es_min(self: "ESNetworkType"):
        src = self.substrate.input_coordinates[0]
        hid = Coordinate(0.25, 0.0)
        w = query_cppn(
            src,
            hid,
            True,
            self.cppn,
            self.max_weight,
            enable_leo=self.enable_leo,
            leo_threshold=self.leo_threshold,
        )
        hidden, conns = set(), set()
        if w != 0.0:
            hidden.add(hid)
            conns.add(Connection3D(src=src, dst=hid, weight=w))
        return hidden, conns

    monkeypatch.setattr(ESNetwork, "es_hyperneat", es_min, raising=False)
    net = ESNetwork(substrate_1in_1out, cppn,
                    params).create_phenotype_network()

    # 断言：出现一个目的节点（隐藏）且有一条入边来自输入
    m = _node_eval_map(net)
    # inputs: [0], outputs: [1], hidden 从 2 开始
    hidden_id = max(m.keys())  # 唯一隐藏
    links = m[hidden_id]
    assert len(links) == 1
    # 避免对“输入一定是 0”的硬编码，直接对比实际映射到的输入索引
    assert links[0][0] == net.input_nodes[0]


# =========================================================
# T3（Deps-1 / D-2）：2D/3D 切换的一致 Schema
# =========================================================
def test_T3_use3d_switch_schema_consistent(monkeypatch, substrate_1in_1out):
    # 统一的 es_hyperneat 替身（忽略 use_3d 分支细节）
    def es_min(self: "ESNetworkType"):
        src = self.substrate.input_coordinates[0]
        hid = Coordinate(0.0, 0.5, 0.0)
        out = self.substrate.output_coordinates[0]
        w1 = 1.0
        w2 = 1.5
        hidden, conns = {
            hid
        }, {Connection3D(src=src, dst=hid, weight=w1), Connection3D(src=hid, dst=out, weight=w2)}
        return hidden, conns

    monkeypatch.setattr(ESNetwork, "es_hyperneat", es_min, raising=False)

    cppn = CPPNStub()
    # 2D
    net2d = ESNetwork(substrate_1in_1out, cppn, make_params(
        use_3d=False)).create_phenotype_network()
    # 3D
    net3d = ESNetwork(substrate_1in_1out, cppn, make_params(
        use_3d=True)).create_phenotype_network()

    # 断言：类型与入/出节点段落一致
    assert isinstance(net2d, neat.nn.RecurrentNetwork)
    assert isinstance(net3d, neat.nn.RecurrentNetwork)
    assert net2d.input_nodes == net3d.input_nodes == [0]
    assert net2d.output_nodes == net3d.output_nodes == [1]


# =========================================================
# T4（Deps-1 / I-3）：band_threshold 边界（<阈值不连 / >阈值连）
# =========================================================
@pytest.mark.parametrize(
    "diff, threshold, expect_edges",
    [
        (0.10, 0.20, 0),  # 低于阈值 → 不连
        (0.30, 0.20, 1),  # 高于阈值 → 连接
    ],
)
def test_T4_band_threshold_boundary(monkeypatch, substrate_1in_1out, diff, threshold, expect_edges):
    params = make_params(band_threshold=threshold)
    cppn = CPPNStub()  # 权值无需真实计算

    def es_band(self: "ESNetworkType"):
        src = self.substrate.input_coordinates[0]
        dst = Coordinate(0.2, 0.2)
        hidden, conns = set(), set()
        if diff > self.band_threshold:
            hidden.add(dst)
            conns.add(Connection3D(src=src, dst=dst, weight=1.0))
        return hidden, conns

    monkeypatch.setattr(ESNetwork, "es_hyperneat", es_band, raising=False)
    net = ESNetwork(substrate_1in_1out, cppn,
                    params).create_phenotype_network()

    # 断言：链接数量符合期望
    m = _node_eval_map(net)
    total_links = sum(len(links) for links in m.values())
    assert total_links == expect_edges


# =========================================================
# T5（Deps-1 / D-4）：三阶段探索合并（去重）
# =========================================================
def test_T5_three_phase_merge_dedup(monkeypatch, substrate_1in_1out):
    cppn = CPPNStub()

    def es_three_phase(self: "ESNetworkType"):
        src = self.substrate.input_coordinates[0]
        h1 = Coordinate(-0.3, 0.3)
        h2 = Coordinate(0.3, 0.3)
        out = self.substrate.output_coordinates[0]
        # 构造可能来自 1/2/3 阶段的重复边；集合应去重
        conns = {
            Connection3D(src=src, dst=h1, weight=1.0),
            Connection3D(src=src, dst=h1, weight=1.0),  # 重复
            Connection3D(src=h1, dst=h2, weight=1.2),
            Connection3D(src=h2, dst=out, weight=1.4),
        }
        hidden = {h1, h2}
        return hidden, conns

    monkeypatch.setattr(ESNetwork, "es_hyperneat",
                        es_three_phase, raising=False)
    net = ESNetwork(substrate_1in_1out, cppn, make_params()
                    ).create_phenotype_network()

    # 断言：目标节点的 links 已天然去重（由 set 去重 + 构造逻辑聚合）
    m = _node_eval_map(net)
    # h1 的入边 1 条，h2 的入边 1 条，out 的入边 1 条 → 共 3
    total_links = sum(len(links) for links in m.values())
    assert total_links == 3


# =========================================================
# T6（Deps-2 / I-1）：索引段连续性（inputs→outputs→hidden）
# =========================================================
def test_T6_index_ranges_continuity(monkeypatch, substrate_2in_2out):
    cppn = CPPNStub()

    def es_min(self: "ESNetworkType"):
        # 一个隐藏点，连接 input0 -> H -> output1
        src0 = self.substrate.input_coordinates[0]
        out1 = self.substrate.output_coordinates[1]
        hid = Coordinate(0.0, 0.0)
        conns = {
            Connection3D(src=src0, dst=hid, weight=1.0),
            Connection3D(src=hid, dst=out1, weight=1.0),
        }
        hidden = {hid}
        return hidden, conns

    monkeypatch.setattr(ESNetwork, "es_hyperneat", es_min, raising=False)
    net = ESNetwork(substrate_2in_2out, cppn, make_params()
                    ).create_phenotype_network()

    # 断言：inputs=[0,1], outputs=[2,3], hidden 从 4 起
    assert net.input_nodes == [0, 1]
    assert net.output_nodes == [2, 3]
    # 找到隐藏节点的 idx（唯一一个非 0/1/2/3 的 node_evals 键）
    hidden_ids = [idx for idx, *_ in net.node_evals if idx not in (0, 1, 2, 3)]
    assert len(hidden_ids) == 1 and hidden_ids[0] >= 4


# =========================================================
# T7（Deps-2 / I-2）：多来源→同一目标的链接归并
# =========================================================
def test_T7_merge_multiple_sources_to_one_dst(monkeypatch, substrate_2in_2out):
    cppn = CPPNStub()

    def es_min(self: "ESNetworkType"):
        i0 = self.substrate.input_coordinates[0]
        i1 = self.substrate.input_coordinates[1]
        o0 = self.substrate.output_coordinates[0]
        # 两条边指向同一输出 o0
        conns = {
            Connection3D(src=i0, dst=o0, weight=0.7),
            Connection3D(src=i1, dst=o0, weight=1.3),
        }
        hidden = set()
        return hidden, conns

    monkeypatch.setattr(ESNetwork, "es_hyperneat", es_min, raising=False)
    net = ESNetwork(substrate_2in_2out, cppn, make_params()
                    ).create_phenotype_network()

    # 断言：o0 的 node_evals 有一条记录，但 links 含两对 (src_id, w)
    dst_idx = net.output_nodes[0]  # o0 的索引
    rec = [ne for ne in net.node_evals if ne[0] == dst_idx][0]
    links = rec[-1]
    assert len(links) == 2
    src_ids = {src for src, _w in links}
    assert src_ids == {0, 1}  # 两个输入


# =========================================================
# T8（Deps-3 / D-1）：clean_net 剔除悬挂边
# =========================================================
def test_T8_clean_net_removes_dangling(substrate_2in_2out):
    cppn = CPPNStub()
    net = ESNetwork(substrate_2in_2out, cppn, make_params())

    i0 = net.substrate.input_coordinates[0]
    o0 = net.substrate.output_coordinates[0]
    h_dangling = Coordinate(-0.2, -0.2)  # 只从输入可达，不可达输出
    h_valid = Coordinate(0.2, 0.2)  # 贯通输入→输出

    conns = {
        Connection3D(src=i0, dst=h_dangling, weight=0.5),  # 将被移除
        Connection3D(src=i0, dst=h_valid, weight=0.5),
        Connection3D(src=h_valid, dst=o0, weight=0.5),
    }

    true_nodes, true_conns = net.clean_net(conns)
    # 断言：悬挂隐藏点被剔除，仅保留贯通路径
    assert h_dangling.to_tuple() not in true_nodes
    assert h_valid.to_tuple() in true_nodes
    assert all(c.src != h_dangling for c in true_conns)


# =========================================================
# T9（Deps-4 / I-1）：node_evals 字段契约
# =========================================================
def test_T9_node_evals_contract(monkeypatch, substrate_1in_1out):
    cppn = CPPNStub()

    def es_min(self: "ESNetworkType"):
        i = self.substrate.input_coordinates[0]
        h = Coordinate(0.0, -0.5)
        conns = {Connection3D(src=i, dst=h, weight=1.0)}
        return {h}, conns

    monkeypatch.setattr(ESNetwork, "es_hyperneat", es_min, raising=False)
    net = ESNetwork(substrate_1in_1out, cppn, make_params()
                    ).create_phenotype_network()

    # 断言：每条 node_eval = (idx, callable, sum, 0.0, 1.0, links)
    for idx, act, agg, bias, resp, links in net.node_evals:
        assert callable(act)
        assert agg is sum
        assert bias == 0.0 and resp == 1.0
        assert isinstance(links, list) and all(
            isinstance(p, tuple) and len(p) == 2 for p in links)


# =========================================================
# T10（Deps-5 / I-1）：RecurrentNetwork 构建冒烟
# =========================================================
def test_T10_recurrent_network_smoke(monkeypatch, substrate_1in_1out):
    cppn = CPPNStub()

    def es_min(self: "ESNetworkType"):
        i = self.substrate.input_coordinates[0]
        o = self.substrate.output_coordinates[0]
        conns = {Connection3D(src=i, dst=o, weight=1.0)}
        return set(), conns

    monkeypatch.setattr(ESNetwork, "es_hyperneat", es_min, raising=False)
    net = ESNetwork(substrate_1in_1out, cppn, make_params()
                    ).create_phenotype_network()

    assert isinstance(net, neat.nn.RecurrentNetwork)
    assert net.input_nodes == [0]
    assert net.output_nodes == [1]
    assert len(net.node_evals) == 1
    idx, *_ = net.node_evals[0]
    assert idx == net.output_nodes[0]


# =========================================================
# A1（整体验收 / I 维黑盒）
# =========================================================
def test_A1_blackbox_acceptance_2d_3d(monkeypatch, substrate_1in_1out, tmp_path):
    # draw_es 冒烟：避免真实落盘副作用
    monkeypatch.setattr(es_mod, "draw_es", lambda *_a,
                        **_k: None, raising=False)

    def es_chain(self: "ESNetworkType"):
        i = self.substrate.input_coordinates[0]
        h = Coordinate(0.4, 0.0)
        o = self.substrate.output_coordinates[0]
        conns = {
            Connection3D(src=i, dst=h, weight=1.1),
            Connection3D(src=h, dst=o, weight=1.2),
        }
        return {h}, conns

    monkeypatch.setattr(ESNetwork, "es_hyperneat", es_chain, raising=False)

    # 2D 验收
    net2d = ESNetwork(
        substrate_1in_1out, CPPNStub(), make_params(
            use_3d=False, enable_leo=True, leo_threshold=0.0)
    ).create_phenotype_network(filename=str(tmp_path / "es2d.png"))
    # 3D 验收
    net3d = ESNetwork(
        substrate_1in_1out, CPPNStub(), make_params(
            use_3d=True, enable_leo=True, leo_threshold=0.0)
    ).create_phenotype_network(filename=str(tmp_path / "es3d.png"))

    for net in (net2d, net3d):
        assert isinstance(net, neat.nn.RecurrentNetwork)
        # Schema：入/出节点序号段落稳定
        assert net.input_nodes == [0]
        assert net.output_nodes == [1]
        # node_evals 至少包含隐藏与输出两条记录
        assert len(net.node_evals) >= 2
