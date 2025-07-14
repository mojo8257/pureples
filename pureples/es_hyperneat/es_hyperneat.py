"""
All logic concerning ES-HyperNEAT resides here.
"""
# 3D_update--------------------------------------------------------------------------------
import copy
import neat
import numpy as np
from pureples.hyperneat.hyperneat import query_cppn
from pureples.shared.visualize import draw_es

# 3D_update
from typing import Tuple
from pureples.shared.coordinate import Coordinate




class QuadPoint:
    """
    Class representing an area in the quadtree.
    Defined by a center coordinate and the distance to the edges of the area.
    """

    def __init__(self, x, y, width, lvl):
        self.x = x
        self.y = y
        self.w = 0.0
        self.width = width
        self.cs = [None] * 4
        self.lvl = lvl

class OctPoint:
    """
    Class representing a region in the octree.
    Defined by a center (x,y,z), half-width, and subdivision level.
    """

    def __init__(self, x: float, y: float, z: float,
                     width: float, lvl: int):
        self.x = x
        self.y = y
        self.z = z
        self.w = 0.0
        self.width = width
        self.cs = [None] * 8  # 八叉树子节点
        self.lvl = lvl


class Connection:
    """
    Class representing a connection from one point to another with a certain weight.
    """

    def __init__(self, x1, y1, x2, y2, weight):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.weight = weight

    # Below is needed for use in set.
    def __eq__(self, other):
        if not isinstance(other, Connection):
            return NotImplemented
        return (self.x1, self.y1, self.x2, self.y2) == (other.x1, other.y1, other.x2, other.y2)

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2, self.weight))


# 3D_update
class Connection3D:
    """
    Represents a connection between two 3D coordinates with a weight.
    """
    def __init__(self,
                 src: Coordinate,
                 dst: Coordinate,
                 weight: float):
        self.src = src
        self.dst = dst
        self.weight = weight

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection3D):
            return False
        return (self.src == other.src and
                self.dst == other.dst and
                self.weight == other.weight)

    def __hash__(self) -> int:
        return hash((self.src.to_tuple(), self.dst.to_tuple(), self.weight))

    def __repr__(self) -> str:
        return f"Connection3D(src={self.src}, dst={self.dst}, weight={self.weight})"


def find_pattern(cppn, coord: Coordinate, res=60, max_weight=5.0):
    im = np.zeros((res, res))
    for x2 in range(res):
        for y2 in range(res):
            x2_scaled = -1.0 + (x2/float(res))*2.0
            y2_scaled = -1.0 + (y2/float(res))*2.0
            # 用 Coordinate 生成目标点，再 to_cppn_input
            dst = Coordinate(x2_scaled, y2_scaled)
            input_vec = coord.to_cppn_input(dst, outgoing=False)
            n = cppn.activate(input_vec)[0]
            im[x2][y2] = n * max_weight
    return im


class ESNetwork:
    """
    The evolvable substrate network.
    """

    def __init__(self, substrate, cppn, params):
        self.substrate = substrate
        self.cppn = cppn
        self.initial_depth = params["initial_depth"]
        self.max_depth = params["max_depth"]
        self.variance_threshold = params["variance_threshold"]
        self.band_threshold = params["band_threshold"]
        self.iteration_level = params["iteration_level"]
        self.division_threshold = params["division_threshold"]
        self.max_weight = params["max_weight"]
        self.connections = set()
        # Number of layers in the network.
        self.activations = 2 ** params["max_depth"] + 1
        activation_functions = neat.activations.ActivationFunctionSet()
        self.activation = activation_functions.get(params["activation"])
        self.enable_leo = params.get("enable_leo", False)
        self.leo_threshold = params.get("leo_threshold", 0.0)
        self.use_3d = params.get("use_3d", False)

    def create_phenotype_network(self, filename=None):
        """
        Create a RecurrentNetwork using the ES-HyperNEAT approach.
        """
        input_coordinates = self.substrate.input_coordinates
        output_coordinates = self.substrate.output_coordinates

        input_nodes = list(range(len(input_coordinates)))
        output_nodes = list(range(len(input_nodes), len(
            input_nodes)+len(output_coordinates)))
        hidden_idx = len(input_coordinates)+len(output_coordinates)

        coordinates, indices, draw_connections, node_evals = [], [], [], []
        nodes = {}

        coordinates.extend(input_coordinates)
        coordinates.extend(output_coordinates)
        indices.extend(input_nodes)
        indices.extend(output_nodes)

        # Map input and output coordinates to their IDs.
        coords_to_id = dict(zip(coordinates, indices))

        # Where the magic happens.
        hidden_nodes, connections = self.es_hyperneat()


        # Map hidden coordinates to their IDs.
        for coord in hidden_nodes:
            coords_to_id[coord] = hidden_idx
            hidden_idx += 1


        # 对每个 Coordinate，检查对应的 Connection3D.dst
        for coord, idx in coords_to_id.items():
            for c in connections:
                if c.dst == coord:              # 如果连接的目的地坐标与当前 coord 相同
                    draw_connections.append(c)
                    src_id = coords_to_id[c.src]
                    if idx in nodes:
                        nodes[idx].append((src_id, c.weight))
                    else:
                        nodes[idx] = [(src_id, c.weight)]
        # --------------------------------------------------------------------------------------

        # Combine the indices with the connections/links;
        # forming node_evals used by the RecurrentNetwork.
        for idx, links in nodes.items():
            node_evals.append((idx, self.activation, sum, 0.0, 1.0, links))

        # Visualize the network?
        if filename is not None:
            draw_es(coords_to_id, draw_connections, filename)

        # This is actually a feedforward network.
        return neat.nn.RecurrentNetwork(input_nodes, output_nodes, node_evals)

    @staticmethod
    def get_weights(p):
        """
        Recursively collect all leaf weights in a QuadPoint or OctPoint.
        """
        temp = []
        def loop(pp):
            if pp is not None and all(child is not None for child in pp.cs):
                # 动态遍历所有子节点，无论是四叉还是八叉
                for child in pp.cs:
                    loop(child)
            else:
                if pp is not None:
                    temp.append(pp.w)
        loop(p)
        return temp

    def variance(self, p):
        """
        Find the variance of a given QuadPoint.
        """
        if not p:
            return 0.0
        return np.var(self.get_weights(p))

    def division_initialization_2d(self, coord, outgoing):
        """
        Initialize the quadtree by dividing it in appropriate quads.
        """
        root = QuadPoint(0.0, 0.0, 1.0, 1)
        q = [root]

        while q:
            p = q.pop(0)

            p.cs[0] = QuadPoint(p.x - p.width/2.0, p.y -
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[1] = QuadPoint(p.x - p.width/2.0, p.y +
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[2] = QuadPoint(p.x + p.width/2.0, p.y +
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[3] = QuadPoint(p.x + p.width/2.0, p.y -
                                p.width/2.0, p.width/2.0, p.lvl + 1)


            for c in p.cs:
                dst = Coordinate(c.x, c.y)
                c.w = query_cppn(coord, dst, outgoing,
                                 self.cppn, self.max_weight, enable_leo=self.enable_leo, leo_threshold=self.leo_threshold)


            if (p.lvl < self.initial_depth) or (p.lvl < self.max_depth and self.variance(p)
                                                > self.division_threshold):
                for child in p.cs:
                    q.append(child)

        return root

    def division_initialization_3d(self, coord, outgoing):
        # 根节点覆盖 [-1,1]^3 立方体
        root = OctPoint(0.0, 0.0, 0.0, 1.0, 1)
        queue = [root]
        while queue:
            p = queue.pop(0)
            # 子立方半边长
            hs = p.width / 2.0
            # 八个象限偏移 (±hs, ±hs, ±hs)
            offsets = [
                (-hs, -hs, -hs), (-hs, -hs, hs),
                (-hs, hs, -hs), (-hs, hs, hs),
                (hs, -hs, -hs), (hs, -hs, hs),
                (hs, hs, -hs), (hs, hs, hs),
            ]
            for i, (dx, dy, dz) in enumerate(offsets):
                c = OctPoint(p.x + dx, p.y + dy, p.z + dz, hs, p.lvl + 1)
                # 计算权重
                dst = Coordinate(c.x, c.y, c.z)
                c.w = query_cppn(
                    coord, dst, outgoing,
                    self.cppn, self.max_weight,
                    enable_leo=self.enable_leo,
                    leo_threshold=self.leo_threshold
                )
                p.cs[i] = c
            # 分割判定同 2D：初始深度或方差阈值
            if (p.lvl < self.initial_depth) or (
                    p.lvl < self.max_depth and self.variance(p) > self.division_threshold):
                queue.extend(p.cs)
        return root


    def pruning_extraction_2d(self, coord, p, outgoing):
        """
        Determines which connections to express - high variance = more connetions.
        """
        for c in p.cs:
            d_left, d_right, d_top, d_bottom = None, None, None, None

            if self.variance(c) > self.variance_threshold:
                self.pruning_extraction_2d(coord, c, outgoing)
            else:
                left = Coordinate(c.x - p.width, c.y)
                right = Coordinate(c.x + p.width, c.y)
                top = Coordinate(c.x, c.y - p.width)
                bottom = Coordinate(c.x, c.y + p.width)

                d_left   = abs(c.w - query_cppn(coord, left,   outgoing,
                                            self.cppn, self.max_weight,
                                            enable_leo=self.enable_leo,
                                            leo_threshold=self.leo_threshold))
                d_right  = abs(c.w - query_cppn(coord, right,  outgoing,
                                            self.cppn, self.max_weight,
                                            enable_leo=self.enable_leo,
                                            leo_threshold=self.leo_threshold))
                d_top    = abs(c.w - query_cppn(coord, top,    outgoing,
                                            self.cppn, self.max_weight,
                                            enable_leo=self.enable_leo,
                                            leo_threshold=self.leo_threshold))
                d_bottom = abs(c.w - query_cppn(coord, bottom, outgoing,
                                            self.cppn, self.max_weight,
                                            enable_leo=self.enable_leo,
                                            leo_threshold=self.leo_threshold))

                con = None
                if max(min(d_top, d_bottom), min(d_left, d_right)) > self.band_threshold:
                    if outgoing:
                        con = Connection(coord.x, coord.y, c.x, c.y, c.w)
                    else:
                        con = Connection(c.x, c.y, coord.x, coord.y, c.w)
                if con is not None:
                    # Nodes will only connect upwards.
                    # If connections to same layer is wanted, change to con.y1 <= con.y2.
                    if not c.w == 0.0 and con.y1 < con.y2 and not (con.x1 == con.x2 and con.y1 == con.y2):
                        self.connections.add(con)

    def pruning_extraction_3d(self, coord, p, outgoing):
        for c in p.cs:
            if self.variance(c) > self.variance_threshold:
                # 3D 分支的递归
                self.pruning_extraction_3d(coord, c, outgoing)
            else:
                # 计算六个相邻方向 (±x, ±y, ±z) 的权重差异
                offsets = [
                    (p.width, 0.0, 0.0), (-p.width, 0.0, 0.0),
                    (0.0, p.width, 0.0), (0.0, -p.width, 0.0),
                    (0.0, 0.0, p.width), (0.0, 0.0, -p.width),
                ]
                diffs = []
                for dx, dy, dz in offsets:
                    neighbor = Coordinate(c.x + dx, c.y + dy, c.z + dz)
                    w_nb = query_cppn(
                        coord, neighbor, outgoing,
                        self.cppn, self.max_weight,
                        enable_leo=self.enable_leo,
                        leo_threshold=self.leo_threshold
                    )
                    diffs.append(abs(c.w - w_nb))

                # 每个轴上取最小差异
                band_x = min(diffs[0], diffs[1])
                band_y = min(diffs[2], diffs[3])
                band_z = min(diffs[4], diffs[5])

                # 任一轴的最小差异超过 band_threshold 则连线
                if max(band_x, band_y, band_z) > self.band_threshold:
                    dst = Coordinate(c.x, c.y, c.z)
                    if outgoing:
                        con = Connection3D(src=coord, dst=dst, weight=c.w)
                    else:
                        con = Connection3D(src=dst, dst=coord, weight=c.w)
                    self.connections.add(con)


    def es_hyperneat(self):
        """
        Explores the hidden nodes and their connections.
        """
        inputs = self.substrate.input_coordinates
        outputs = self.substrate.output_coordinates
        hidden_nodes: set[tuple[float, float, float]] = set()  # 发现过的所有隐藏节点
        unexplored_hidden_nodes: set[tuple[float, float, float]] = set()  # “待办”队列
        explored_hidden_nodes: set[tuple[float, float, float]] = set()  # ★ 新增：历史已探节点
        connections1, connections2, connections3 = set(), set(), set()

        # 第一阶段：从输入节点探索
        for coord in inputs:

            if self.use_3d:
                root = self.division_initialization_3d(coord, True)
            else:
                root = self.division_initialization_2d(coord, True)

            if self.use_3d:
                self.pruning_extraction_3d(coord, root, True)
            else:
                self.pruning_extraction_2d(coord, root, True)


            connections1 |= self.connections
            for c in self.connections:
                # c.dst 是 Coordinate，对应旧 c.x2, c.y2
                hidden_nodes.add(c.dst.to_tuple())
            self.connections.clear()

        unexplored_hidden_nodes = hidden_nodes - explored_hidden_nodes

        # 第二阶段：迭代探索隐藏节点
        for _ in range(self.iteration_level):
            if not unexplored_hidden_nodes:
                break

            for node in unexplored_hidden_nodes:
                # node 是 (x, y, z)
                explored_hidden_nodes.add(node)
                coord = Coordinate(*node)


                if self.use_3d:
                    root = self.division_initialization_3d(coord, True)
                else:
                    root = self.division_initialization_2d(coord, True)

                if self.use_3d:
                    self.pruning_extraction_3d(coord, root, True)
                else:
                    self.pruning_extraction_2d(coord, root, True)


                connections2 |= self.connections
                for c in self.connections:
                    hidden_nodes.add(c.dst.to_tuple())
                self.connections.clear()

            unexplored_hidden_nodes = hidden_nodes - explored_hidden_nodes

        # 第三阶段：从输出节点反向探索
        for coord in outputs:

            if self.use_3d:
                root = self.division_initialization_3d(coord, False)
            else:
                root = self.division_initialization_2d(coord, False)

            if self.use_3d:
                self.pruning_extraction_3d(coord, root, False)
            else:
                self.pruning_extraction_2d(coord, root, False)


            connections3 |= self.connections
            self.connections.clear()

        all_connections = connections1 | connections2 | connections3
        return self.clean_net(all_connections)

    def clean_net(self, connections: set[Connection3D]):
        """
        Clean a net for dangling connections:
        Intersects paths from input nodes with paths to output.
        """
        # 1) substrate.input/output_coordinates 已经是 List[Coordinate]
        connected_to_inputs = set(coord.to_tuple()
                                  for coord in self.substrate.input_coordinates)
        connected_to_outputs = set(coord.to_tuple()
                                   for coord in self.substrate.output_coordinates)
        true_connections: set[Connection3D] = set()

        initial_input_connections = copy.deepcopy(connections)
        initial_output_connections = copy.deepcopy(connections)

        # 2) 从 inputs 向外扩散
        add_happened = True
        while add_happened:
            add_happened = False
            for c in list(initial_input_connections):
                # 使用 Connection3D.src / dst
                src_tuple = c.src.to_tuple()
                dst_tuple = c.dst.to_tuple()
                if src_tuple in connected_to_inputs:
                    connected_to_inputs.add(dst_tuple)
                    initial_input_connections.remove(c)
                    add_happened = True

        # 3) 从 outputs 向内扩散
        add_happened = True
        while add_happened:
            add_happened = False
            for c in list(initial_output_connections):
                src_tuple = c.src.to_tuple()
                dst_tuple = c.dst.to_tuple()
                if dst_tuple in connected_to_outputs:
                    connected_to_outputs.add(src_tuple)
                    initial_output_connections.remove(c)
                    add_happened = True

        # 4) 交集为真正连通的节点
        true_nodes = connected_to_inputs.intersection(connected_to_outputs)

        # 5) 过滤有效连接
        for c in connections:
            if c.src.to_tuple() in true_nodes and c.dst.to_tuple() in true_nodes:
                true_connections.add(c)

        # 6) 去除输入/输出节点本身
        in_out_tuples = set(coord.to_tuple() for coord in
                            (self.substrate.input_coordinates +
                             self.substrate.output_coordinates))
        true_nodes -= in_out_tuples

        return true_nodes, true_connections

def get_nodes_and_edges(self):
    """
    Returns:
    nodes: List[Coordinate] of hidden nodes (excluding input/output)
    edges: List[Connection3D] of all expressed 3D connections
    """
    node_tuples, conns = self.es_hyperneat()
    # 将三元组重装回 Coordinate
    nodes = [Coordinate(x, y, z) for (x, y, z) in node_tuples]
    return nodes, list(conns)
# 3D_update--------------------------------------------------------------------------------

