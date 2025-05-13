"""
All Hyperneat related logic resides here.
"""

import neat


def create_phenotype_network(cppn, substrate, activation_function="sigmoid",  enable_leo=False, leo_threshold=0.0):
    """
    Creates a recurrent network using a cppn and a substrate.
    """
    input_coordinates = substrate.input_coordinates
    output_coordinates = substrate.output_coordinates
    # List of layers, first index = top layer.
    hidden_coordinates = substrate.hidden_coordinates

    input_nodes = list(range(len(input_coordinates)))
    output_nodes = list(range(len(input_nodes), len(
        input_nodes)+len(output_coordinates)))

    counter = 0
    for layer in hidden_coordinates:
        counter += len(layer)

    hidden_nodes = range(len(input_nodes)+len(output_nodes),
                         len(input_nodes)+len(output_nodes)+counter)

    node_evals = []

    # Get activation function.
    activation_functions = neat.activations.ActivationFunctionSet()
    activation = activation_functions.get(activation_function)

    # Connect hidden to output.
    counter = 0
    for oc in output_coordinates:
        idx = 0
        for layer in hidden_coordinates:
            im = find_neurons(cppn, oc, layer, hidden_nodes[idx], False, enable_leo=enable_leo, leo_threshold=leo_threshold)
            idx += len(layer)
            if im:
                node_evals.append(
                    (output_nodes[counter], activation, sum, 0.0, 1.0, im))

        counter += 1

    # Connect hidden to hidden - starting from the top layer.
    current_layer = 1
    idx = 0
    for layer in hidden_coordinates:
        idx += len(layer)
        counter = idx - len(layer)
        for i in range(current_layer, len(hidden_coordinates)):
            for hc in layer:
                im = find_neurons(
                    cppn, hc, hidden_coordinates[i], hidden_nodes[idx], False, enable_leo=enable_leo, leo_threshold=leo_threshold)
                if im:
                    node_evals.append(
                        (hidden_nodes[counter], activation, sum, 0.0, 1.0, im))
                counter += 1

            counter -= idx

        current_layer += 1

    # Connect input to hidden.
    counter = 0
    for layer in hidden_coordinates:
        for hc in layer:
            im = find_neurons(cppn, hc, input_coordinates,
                              input_nodes[0], False, enable_leo=enable_leo, leo_threshold=leo_threshold)
            if im:
                node_evals.append(
                    (hidden_nodes[counter], activation, sum, 0.0, 1.0, im))
            counter += 1

    return neat.nn.RecurrentNetwork(input_nodes, output_nodes, node_evals)


def find_neurons(cppn, coord, nodes, start_idx, outgoing, max_weight=5.0,
                 enable_leo=False, leo_threshold=0.0):
    """
    Find the neurons to which the given coord is connected.
    """
    im = []
    idx = start_idx

    for node in nodes:
        w = query_cppn(coord, node, outgoing, cppn, max_weight,
                       enable_leo=enable_leo, leo_threshold=leo_threshold)

        if w != 0.0:  # Only include connection if the weight isn't 0.0.
            im.append((idx, w))
        idx += 1

    return im


def query_cppn(coord_src, coord_dst, outgoing, cppn,
               max_weight=5.0, enable_leo=False, leo_threshold=0.0):
    """
    向 CPPN 查询连线权重 (outputs[0]) 以及可选的 LEO (outputs[1]).
    - 若 enable_leo=True 且 CPPN 拥有 >=2 输出，按 LEO 判断是否表达；
    - 否则回退到单输出逻辑。
    """
    if outgoing:
        cppn_input = [coord_src[0], coord_src[1], coord_dst[0], coord_dst[1], 1.0]
    else:
        cppn_input = [coord_dst[0], coord_dst[1], coord_src[0], coord_src[1], 1.0]

    o = cppn.activate(cppn_input)

    # ---- 单/双输出统一处理 ----
    w_raw = o[0]
    leo_val = o[1] if enable_leo and len(o) > 1 else None

    # 1) 若使用 LEO 且其值低于阈值 ⇒ 不表达
    if leo_val is not None and leo_val < leo_threshold:
        return 0.0

    # 2) 再看权值本身是否越过 ±0.2 的 dead-zone
    if abs(w_raw) <= 0.2:
        return 0.0

    # 3) 线性压缩到 [-max_weight, max_weight]
    w = (w_raw - 0.2) / 0.8 if w_raw > 0 else (w_raw + 0.2) / 0.8
    return w * max_weight