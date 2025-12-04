# tests/test_b21_cppn_create.py
# -*- coding: utf-8 -*-
"""
B2.1 CPPN 构建（genome→CPPN）集成测试（聚焦 LEO / 局部种子 / 3D 输入切换）

修正点：
1) 为 Dummy 连接基因补充 .key（通过 ConnectionDict 在 __setitem__ 自动写入），
   对齐 NEAT 的 ConnectionGene 契约，修复 create() 首次读取 connections 时的崩溃。
2) T7 增加 input->H 入边，保证层序可达（否则 H/LEO 不会出现在 layers）。
3) A1 的类型断言改为 isinstance(net, FeedForwardNetwork)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Any, Optional

import pytest


# ========= SUT 加载 =========
def _import_sut() -> Any:
    """
    返回 neat.nn.FeedForwardNetwork 类（含 @staticmethod create）
    """
    from neat.nn import FeedForwardNetwork  # type: ignore
    return FeedForwardNetwork


FeedForwardNetwork = _import_sut()


# ========= 桩实现（最小接口） =========
# FeedForwardNetwork.create(genome, config) 只会用到如下接口：
# - genome.connections: Dict[(inode,onode)]->ConnGene(weight, enabled, key)
# - genome.nodes: Dict[node_id]->NodeGene(activation, aggregation, bias, response)
# - config.genome_config.input_keys / output_keys
# - config.genome_config.activation_defs.get(name) / aggregation_function_defs.get(name)

@dataclass
class DummyConnGene:
    # key 由 ConnectionDict 在写入时自动回填
    key: Optional[Tuple[int, int]] = None
    weight: float = 1.0
    enabled: bool = True


@dataclass
class DummyNodeGene:
    activation: str = "sigmoid"
    aggregation: str = "sum"
    bias: float = 0.0
    response: float = 1.0


class ConnectionDict(dict):
    """让 connections 的 value 自动拥有 .key，与 NEAT 的 ConnectionGene 对齐"""
    def __setitem__(self, k, v):
        # k 是 (inode, onode)
        if not hasattr(v, "key") or v.key != k:
            try:
                v.key = k
            except Exception:
                object.__setattr__(v, "key", k)
        super().__setitem__(k, v)


class DummyGenome:
    def __init__(self):
        self.connections: Dict[Tuple[int, int], DummyConnGene] = ConnectionDict()
        self.nodes: Dict[int, DummyNodeGene] = {}


class _Namespace:
    """简易命名空间容器"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _act_sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _act_step(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


def _act_gauss(x: float) -> float:
    return math.exp(-x * x)


def _agg_sum(xs: List[float]) -> float:
    return sum(xs)


def make_defs():
    # 用“同一对象”的函数表，便于用 is 进行断言
    activation_defs = {
        "sigmoid": _act_sigmoid,
        "step": _act_step,
        "gauss": _act_gauss,
    }
    aggregation_defs = {
        "sum": _agg_sum,
    }
    return activation_defs, aggregation_defs


def make_config(input_keys: List[int], output_keys: List[int]) -> Any:
    act_defs, agg_defs = make_defs()
    genome_config = _Namespace(
        input_keys=input_keys,
        output_keys=output_keys,
        activation_defs=act_defs,
        aggregation_function_defs=agg_defs,
    )
    return _Namespace(genome_config=genome_config)


# ========= 测试夹具 =========

@pytest.fixture
def base_inputs_2d():
    # 2D 模式：5 个输入（使用负索引，避免与节点/输出冲突）
    return [-1, -2, -3, -4, -5]


@pytest.fixture
def base_inputs_3d():
    # 3D 模式：7 个输入
    return [-1, -2, -3, -4, -5, -6, -7]


@pytest.fixture
def keys():
    # 约定：主输出=0，LEO=1，隐藏示例=100
    return {"OUT": 0, "LEO": 1, "H": 100}


# ========= T1：LEO 关闭基线 =========
def test_T1_leo_off_baseline(base_inputs_2d, keys):
    g = DummyGenome()
    OUT = keys["OUT"]
    # 输出节点
    g.nodes[OUT] = DummyNodeGene(activation="sigmoid", aggregation="sum", bias=0.0, response=1.0)
    # 一条可用连接：(-1)->0
    g.connections[(base_inputs_2d[0], OUT)] = DummyConnGene(weight=1.0, enabled=True)

    cfg = make_config(input_keys=base_inputs_2d, output_keys=[OUT])

    net = FeedForwardNetwork.create(g, cfg)

    assert net.output_nodes == [OUT]
    # node_evals 里必须有输出 0 的记录，且激活函数表映射正确
    rec = [ne for ne in net.node_evals if ne[0] == OUT]
    assert len(rec) == 1
    node, act_fn, agg_fn, bias, resp, inputs = rec[0]
    assert act_fn is cfg.genome_config.activation_defs["sigmoid"]
    assert agg_fn is cfg.genome_config.aggregation_function_defs["sum"]
    assert inputs == [(base_inputs_2d[0], 1.0)]


# ========= T2：LEO 打开但无入边，不入层 =========
def test_T2_leo_on_no_edges_excluded(base_inputs_2d, keys):
    OUT, LEO = keys["OUT"], keys["LEO"]
    g = DummyGenome()
    g.nodes[OUT] = DummyNodeGene(activation="sigmoid", aggregation="sum")
    # 仅主输出有一条入边
    g.connections[(base_inputs_2d[0], OUT)] = DummyConnGene(weight=0.5, enabled=True)

    # LEO 开启且位于输出末尾，但不给入边
    cfg = make_config(input_keys=base_inputs_2d, output_keys=[OUT, LEO])

    net = FeedForwardNetwork.create(g, cfg)

    assert net.output_nodes == [OUT, LEO]  # 顺序：LEO 在末尾
    # LEO 无入边，应不出现在 node_evals
    assert all(ne[0] != LEO for ne in net.node_evals)
    # 主输出仍正常
    assert any(ne[0] == OUT for ne in net.node_evals)


# ========= T3：xaxis 局部种子等价结构（gauss 隐藏 → step LEO） =========
def test_T3_xaxis_equivalent_graph(base_inputs_2d, keys):
    OUT, LEO, H = keys["OUT"], keys["LEO"], keys["H"]
    g = DummyGenome()
    # 隐藏节点：gauss
    g.nodes[H] = DummyNodeGene(activation="gauss", aggregation="sum")
    # LEO：step
    g.nodes[LEO] = DummyNodeGene(activation="step", aggregation="sum", bias=0.3, response=1.0)
    # 连接：x1(+1), x2(-1) → H → LEO(+1)
    g.connections[(base_inputs_2d[0], H)] = DummyConnGene(weight=+1.0, enabled=True)
    g.connections[(base_inputs_2d[1], H)] = DummyConnGene(weight=-1.0, enabled=True)
    g.connections[(H, LEO)] = DummyConnGene(weight=+1.0, enabled=True)

    # 输出包含主输出与 LEO（主输出可无入边不参与计算，本测试仅关注 LEO 路径）
    cfg = make_config(input_keys=base_inputs_2d, output_keys=[OUT, LEO])
    net = FeedForwardNetwork.create(g, cfg)

    # LEO 出现在评估序列中，激活函数映射为 step
    leo_rec = [ne for ne in net.node_evals if ne[0] == LEO]
    assert len(leo_rec) == 1
    _, act_fn, agg_fn, bias, resp, inputs = leo_rec[0]
    assert act_fn is cfg.genome_config.activation_defs["step"]
    # 隐藏节点也应出现，且在 LEO 之前（分层顺序）
    order = [ne[0] for ne in net.node_evals]
    assert H in order and LEO in order and order.index(H) < order.index(LEO)


# ========= T4：忽略禁用连接 =========
def test_T4_ignore_disabled_edges(base_inputs_2d, keys):
    OUT = keys["OUT"]
    g = DummyGenome()
    g.nodes[OUT] = DummyNodeGene(activation="sigmoid", aggregation="sum")
    g.connections[(base_inputs_2d[0], OUT)] = DummyConnGene(weight=1.0, enabled=True)
    g.connections[(base_inputs_2d[1], OUT)] = DummyConnGene(weight=2.0, enabled=False)  # 禁用
    cfg = make_config(input_keys=base_inputs_2d, output_keys=[OUT])

    net = FeedForwardNetwork.create(g, cfg)
    rec = [ne for ne in net.node_evals if ne[0] == OUT][0]
    inputs = rec[-1]
    assert inputs == [(base_inputs_2d[0], 1.0)]  # 仅保留启用边


# ========= T5：两层链路顺序（input→H→OUT） =========
def test_T5_two_layer_order(base_inputs_2d, keys):
    OUT, H = keys["OUT"], keys["H"]
    g = DummyGenome()
    g.nodes[H] = DummyNodeGene(activation="sigmoid", aggregation="sum")
    g.nodes[OUT] = DummyNodeGene(activation="sigmoid", aggregation="sum")
    g.connections[(base_inputs_2d[0], H)] = DummyConnGene(weight=1.0, enabled=True)
    g.connections[(H, OUT)] = DummyConnGene(weight=1.0, enabled=True)
    cfg = make_config(input_keys=base_inputs_2d, output_keys=[OUT])

    net = FeedForwardNetwork.create(g, cfg)
    order = [ne[0] for ne in net.node_evals]
    assert order.index(H) < order.index(OUT)


# ========= T6：多入边聚合函数绑定 =========
def test_T6_multi_input_aggregation_binding(base_inputs_2d, keys):
    OUT = keys["OUT"]
    g = DummyGenome()
    g.nodes[OUT] = DummyNodeGene(activation="sigmoid", aggregation="sum")
    # 两条入边
    g.connections[(base_inputs_2d[0], OUT)] = DummyConnGene(weight=1.0, enabled=True)
    g.connections[(base_inputs_2d[1], OUT)] = DummyConnGene(weight=1.5, enabled=True)
    cfg = make_config(input_keys=base_inputs_2d, output_keys=[OUT])

    net = FeedForwardNetwork.create(g, cfg)
    node, act_fn, agg_fn, bias, resp, inputs = [ne for ne in net.node_evals if ne[0] == OUT][0]
    assert agg_fn is cfg.genome_config.aggregation_function_defs["sum"]
    # 同时验证两条入边保留
    assert set(inputs) == {(base_inputs_2d[0], 1.0), (base_inputs_2d[1], 1.5)}


# ========= T7：LEO 激活函数映射一致性（补充 input->H 入边确保可达） =========
def test_T7_leo_activation_binding(base_inputs_2d, keys):
    LEO, H = keys["LEO"], keys["H"]
    g = DummyGenome()
    g.nodes[H] = DummyNodeGene(activation="sigmoid", aggregation="sum")
    g.nodes[LEO] = DummyNodeGene(activation="step", aggregation="sum", bias=0.0, response=1.0)
    # 关键：补充输入到 H 的入边，保证层序可达
    g.connections[(base_inputs_2d[0], H)] = DummyConnGene(weight=1.0, enabled=True)
    g.connections[(H, LEO)] = DummyConnGene(weight=1.0, enabled=True)
    cfg = make_config(input_keys=base_inputs_2d, output_keys=[LEO])

    net = FeedForwardNetwork.create(g, cfg)
    leo_rec = [ne for ne in net.node_evals if ne[0] == LEO][0]
    assert leo_rec[1] is cfg.genome_config.activation_defs["step"]  # activation_function 绑定


# ========= T8：bias/response 透传 =========
def test_T8_bias_response_passthrough(base_inputs_2d, keys):
    OUT = keys["OUT"]
    g = DummyGenome()
    g.nodes[OUT] = DummyNodeGene(activation="sigmoid", aggregation="sum", bias=0.42, response=0.73)
    g.connections[(base_inputs_2d[2], OUT)] = DummyConnGene(weight=0.9, enabled=True)
    cfg = make_config(input_keys=base_inputs_2d, output_keys=[OUT])

    net = FeedForwardNetwork.create(g, cfg)
    _, _, _, bias, resp, _ = [ne for ne in net.node_evals if ne[0] == OUT][0]
    assert bias == 0.42 and resp == 0.73


# ========= T9：3D 输入键同步（5→7） =========
def test_T9_input_keys_3d_switch(base_inputs_2d, base_inputs_3d, keys):
    OUT = keys["OUT"]
    # 2D
    g2d = DummyGenome()
    g2d.nodes[OUT] = DummyNodeGene(activation="sigmoid", aggregation="sum")
    g2d.connections[(base_inputs_2d[0], OUT)] = DummyConnGene(weight=1.0, enabled=True)
    cfg2d = make_config(input_keys=base_inputs_2d, output_keys=[OUT])
    net2d = FeedForwardNetwork.create(g2d, cfg2d)
    assert net2d.input_nodes == base_inputs_2d and len(net2d.input_nodes) == 5

    # 3D
    g3d = DummyGenome()
    g3d.nodes[OUT] = DummyNodeGene(activation="sigmoid", aggregation="sum")
    g3d.connections[(base_inputs_3d[0], OUT)] = DummyConnGene(weight=1.0, enabled=True)
    cfg3d = make_config(input_keys=base_inputs_3d, output_keys=[OUT])
    net3d = FeedForwardNetwork.create(g3d, cfg3d)
    assert net3d.input_nodes == base_inputs_3d and len(net3d.input_nodes) == 7


# ========= A1：整体验收（仅 I 维黑盒） =========
def test_A1_blackbox_acceptance(base_inputs_2d, keys):
    OUT, LEO, H = keys["OUT"], keys["LEO"], keys["H"]
    # 构造一条到主输出的路径 + 一条到 LEO 的路径，保证 node_evals 非空且覆盖两类输出
    g = DummyGenome()
    # 主输出路径：I0 -> H -> OUT
    g.nodes[H] = DummyNodeGene(activation="sigmoid", aggregation="sum")
    g.nodes[OUT] = DummyNodeGene(activation="sigmoid", aggregation="sum")
    g.connections[(base_inputs_2d[0], H)] = DummyConnGene(weight=1.0, enabled=True)
    g.connections[(H, OUT)] = DummyConnGene(weight=1.0, enabled=True)
    # LEO 路径：I1 -> LEO（step）
    g.nodes[LEO] = DummyNodeGene(activation="step", aggregation="sum", bias=0.1, response=1.0)
    g.connections[(base_inputs_2d[1], LEO)] = DummyConnGene(weight=0.8, enabled=True)

    cfg = make_config(input_keys=base_inputs_2d, output_keys=[OUT, LEO])

    # 快照（无副作用）
    snap_nodes = {k: (v.activation, v.aggregation, v.bias, v.response) for k, v in g.nodes.items()}
    snap_conns = {k: (v.weight, v.enabled) for k, v in g.connections.items()}

    net = FeedForwardNetwork.create(g, cfg)

    # I 维黑盒断言
    assert isinstance(net, FeedForwardNetwork), "返回对象必须是 FeedForwardNetwork 实例"
    assert net.input_nodes == cfg.genome_config.input_keys
    assert net.output_nodes == cfg.genome_config.output_keys
    assert len(net.node_evals) >= 2  # 至少包含 OUT/LEO 的计算节点或其上游

    # 无副作用：genome 不应被改写
    assert snap_nodes == {k: (v.activation, v.aggregation, v.bias, v.response) for k, v in g.nodes.items()}
    assert snap_conns == {k: (v.weight, v.enabled) for k, v in g.connections.items()}
