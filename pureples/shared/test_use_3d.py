# tests/test_esnetwork_use_3d.py

import pytest
from pureples.es_hyperneat.es_hyperneat import ESNetwork   # ESNetwork 定义于 es_hyperneat.py :contentReference[oaicite:0]{index=0}
from pureples.shared.substrate  import Substrate      # Substrate 定义于 substrate.py :contentReference[oaicite:1]{index=1}

class DummyCPPN:
    """桩对象：不需要实际行为，只需能被 ESNetwork 存储"""
    pass

# 构造一个包含 __init__ 所需最小合法参数的字典
BASE_PARAMS = {
    "initial_depth":       1,
    "max_depth":           1,
    "variance_threshold":  0.0,
    "band_threshold":      0.0,
    "iteration_level":     1,
    "division_threshold":  0.0,
    "max_weight":          1.0,
    "activation":         "sigmoid"
}

def test_use_3d_explicit_true():
    """
    当 params 中显式包含 use_3d=True 时，
    ESNetwork.use_3d 应被设置为 True。
    """
    params = {**BASE_PARAMS, "use_3d": True}
    substrate = Substrate([], [])      # 不关心具体坐标，只要属性存在
    cppn = DummyCPPN()
    es = ESNetwork(substrate, cppn, params)

    assert es.use_3d is True

def test_use_3d_default_false():
    """
    当 params 中缺省 use_3d 时，
    ESNetwork.use_3d 应使用默认值 False。
    """
    params = BASE_PARAMS.copy()        # 不包含 "use_3d"
    substrate = Substrate([], [])
    cppn = DummyCPPN()
    es = ESNetwork(substrate, cppn, params)

    assert es.use_3d is False
