"""
Varying visualisation tools.
"""

import pickle
import graphviz
import matplotlib.pyplot as plt
# 3D_update--------------------------------------------------------------------------------
import plotly.graph_objects as go
# 3D_update--------------------------------------------------------------------------------
from typing import List, Tuple
from pureples.shared.coordinate import Coordinate

def draw_net(net, filename=None, node_names={}, node_colors={}):
    """
    Draw neural network with arbitrary topology.
    """
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph('svg', node_attr=node_attrs)

    inputs = set()
    for k in net.input_nodes:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box',
                       'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in net.output_nodes:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled',
                      'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    for node, _, _, _, _, links in net.node_evals:
        for i, w in links:
            node_input, output = node, i
            a = node_names.get(output, str(output))
            b = node_names.get(node_input, str(node_input))
            style = 'solid'
            color = 'green' if w > 0.0 else 'red'
            width = str(0.1 + abs(w / 5.0))
            dot.edge(a, b, _attributes={
                     'style': style, 'color': color, 'penwidth': width})

    dot.render(filename)

    return dot


# 3D_update--------------------------------------------------------------------------------
def draw_network_3d(nodes: List[Coordinate],edges: List[Tuple[Coordinate, Coordinate]],title: str = None) -> go.Figure:
    """
    使用 Plotly 绘制交互式 3D 网络：
      - nodes: Coordinate 列表
      - edges: (src, dst) 二元组列表
    返回一个 go.Figure，可 .show() 或 .write_html()
    """
    fig = go.Figure()
    # 节点散点
    fig.add_trace(go.Scatter3d(
        x=[n.x for n in nodes],
        y=[n.y for n in nodes],
        z=[n.z for n in nodes],
        mode="markers",
        marker=dict(size=4),
        hovertext=[f"{n.x:.2f}, {n.y:.2f}, {n.z:.2f}" for n in nodes]
    ))
    # 连线
    for src, dst in edges:
        fig.add_trace(go.Scatter3d(
            x=[src.x, dst.x],
            y=[src.y, dst.y],
            z=[src.z, dst.z],
            mode="lines"
        ))
    if title:
        fig.update_layout(title=title)
    return fig
# 3D_update--------------------------------------------------------------------------------


def onclick(event):
    """
    Click handler for weight gradient created by a CPPN. Will re-query with the clicked coordinate.
    """
    plt.close()
    x = event.xdata
    y = event.ydata

    path_to_cppn = "es_hyperneat_xor_small_cppn.pkl"
    # For now, path_to_cppn should match path in test_cppn.py, sorry.
    with open(path_to_cppn, 'rb') as cppn_input:
        cppn = pickle.load(cppn_input)
        from pureples.es_hyperneat.es_hyperneat import find_pattern
        pattern = find_pattern(cppn, (x, y))
        draw_pattern(pattern)


def draw_pattern(im, res=60):
    """
    Draws the pattern/weight gradient queried by a CPPN.
    """
    fig = plt.figure()
    plt.axis([-1, 1, -1, 1])
    fig.add_subplot(111)

    a = range(res)
    b = range(res)

    for x in a:
        for y in b:
            px = -1.0 + (x/float(res))*2.0+1.0/float(res)
            py = -1.0 + (y/float(res))*2.0+1.0/float(res)
            c = str(0.5-im[x][y]/float(res))
            plt.plot(px, py, marker='s', color=c)

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.grid()
    plt.show()


def draw_es(id_to_coords, connections, filename):
    """
    Draw the net created by ES-HyperNEAT
    """
    fig = plt.figure()
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    fig.add_subplot(111)

    for c in connections:
        color = 'red'
        if c.weight > 0.0:
            color = 'black'
        plt.arrow(c.x1, c.y1, c.x2-c.x1, c.y2-c.y1, head_width=0.00, head_length=0.0,
                  fc=color, ec=color, length_includes_head=True)

    for (coord, _) in id_to_coords.items():
        plt.plot(coord[0], coord[1], marker='o', markersize=8.0, color='grey')

    plt.grid()
    fig.savefig(filename)
