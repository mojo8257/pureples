# tests/test_cli_flags.py

import sys
import os
import subprocess
import runpy
import pytest
import types
from pathlib import Path

from pureples.es_hyperneat.es_hyperneat import ESNetwork
import pureples.experiments.retina.es_hyperneat_retina as retina

@pytest.mark.parametrize("cmd,args,expected", [
    # 场景 A-1：显式开启 --use-3d，应传递 use_3d=True 给 ESNetwork.__init__
    ("use3d_on", ["--use-3d"], True),
    # 场景 A-2：不传 --use-3d，use_3d 应为 False 或未设置
    ("use3d_off", [], False),
    # 场景 A-3：重复传递 --use-3d，应正常退出且只识别一次 True
    ("use3d_repeat", ["--use-3d", "--use-3d"], True),
])
def test_use_3d_flag(tmp_path, monkeypatch, cmd, args, expected):
    """
    A: 验证脚本层的 --use-3d 参数能够正确透传到 ESNetwork.__init__ 中。
    - 替换 ESNetwork.__init__，记录其接收到的 use_3d 值。
    - 通过 subprocess 以模块方式执行脚本，传入不同的 args。
    """
    captured = {}

    def fake_init(self, substrate, cppn, params):
        # 记录 params 中 use_3d 字段
        captured['use_3d'] = params.get("use_3d", False)
        # 不调用原始构造，直接退出以免后续依赖失败
    monkeypatch.setattr(ESNetwork, "__init__", fake_init)

    # 设置 PYTHONPATH，确保模块能被 -m 方式找到
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)  # 根据项目结构调整

    # 调用脚本
    result = subprocess.run(
        [sys.executable, "-m", "pureples.experiments.retina.es_hyperneat_retina", *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # 脚本应正常退出
    assert result.returncode == 0, f"stderr:\n{result.stderr.decode()}"
    # 验证 use_3d 传递是否符合预期
    assert captured.get("use_3d", False) is expected


@pytest.mark.parametrize("cmd,args,expected", [
    # 场景 B-1：传入 --generations 50，应传 50
    ("gen_cover", ["--generations", "50"], 50),
    # 场景 B-2：不传，使用源码默认值 2000
    ("gen_default", [], 2000),
])
def test_generations_flag(tmp_path, monkeypatch, cmd, args, expected):
    """
    B: 验证 --generations 参数能够覆盖或保持默认，并传递给 run()。
    - 替换 retina.run 为桩函数，记录接收到的 generations 参数后退出。
    - 使用 runpy.run_module 以避免子进程复杂度，直接在当前进程执行脚本。
    """
    captured = {}

    def fake_run(generations=2000):
        captured['generations'] = generations
        # 模拟正常退出
        sys.exit(0)

    monkeypatch.setattr(retina, "run", fake_run)

    # 模拟命令行参数
    monkeypatch.setattr(sys, "argv", ["es_hyperneat_retina.py", *args])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("pureples.experiments.retina.es_hyperneat_retina", run_name="__main__")
    assert exc.value.code == 0
    assert captured["generations"] == expected


def test_generations_invalid(tmp_path, monkeypatch):
    """
    B-3: 非法值 --generations -1，应以非零退出码结束，stderr 包含提示信息。
    """
    # 直接通过 subprocess 调用，依赖 argparse 的报错
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path.parent)
    result = subprocess.run(
        [sys.executable, "-m", "pureples.experiments.retina.es_hyperneat_retina", "--generations", "-1"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # 非零退出
    assert result.returncode != 0
    err = result.stderr.decode().lower()
    assert "value must be positive" in err or "invalid" in err


def test_combined_flags(tmp_path, monkeypatch):
    """
    C-1 & C-2: 同时传 --use-3d 和 --generations 10，且顺序无关。
    验证两者互不干扰，分别到达 ESNetwork.__init__ 和 run()。
    """
    captured_init = {}
    captured_run = {}

    def fake_init(self, substrate, cppn, params):
        captured_init['use_3d'] = params.get("use_3d", False)

    def fake_run(generations=2000):
        captured_run['generations'] = generations
        sys.exit(0)

    monkeypatch.setattr(ESNetwork, "__init__", fake_init)
    monkeypatch.setattr(retina, "run", fake_run)

    for args in (["--use-3d", "--generations", "10"], ["--generations", "10", "--use-3d"]):
        # 清空记录
        captured_init.clear()
        captured_run.clear()

        monkeypatch.setattr(sys, "argv", ["es_hyperneat_retina.py", *args])
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("pureples.experiments.retina.es_hyperneat_retina", run_name="__main__")
        assert exc.value.code == 0

        assert captured_init.get("use_3d", False) is True
        assert captured_run.get("generations", None) == 10
