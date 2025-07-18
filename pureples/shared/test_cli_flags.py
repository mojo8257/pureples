# tests/test_cli_flags.py

import sys
import os
import subprocess
import pytest
from pathlib import Path

from pureples.es_hyperneat.es_hyperneat import ESNetwork
import pureples.experiments.retina.es_hyperneat_retina as retina


@pytest.mark.parametrize("cmd,args,expected", [
    ("use3d_on", ["--use-3d"], True),
    ("use3d_off", [], False),
    ("use3d_repeat", ["--use-3d", "--use-3d"], True),
])
def test_use_3d_flag(tmp_path, monkeypatch, cmd, args, expected):
    """
    A: 验证脚本层的 --use-3d 参数能够正确透传到 ESNetwork.__init__ 中。
    - 用 monkeypatch 替换 ESNetwork.__init__，记录 params["use_3d"]。
    - 直接调用 main() 并捕获 SystemExit。
    """
    captured = {}

    def fake_init(self, substrate, cppn, params):
        captured['use_3d'] = params.get("use_3d", False)
        sys.exit(0)

    monkeypatch.setattr(ESNetwork, "__init__", fake_init)
    monkeypatch.setattr(sys, "argv", ["es_hyperneat_retina.py", *args])

    with pytest.raises(SystemExit) as exc:
        retina.main()
    assert exc.value.code == 0
    assert captured.get("use_3d", False) is expected


@pytest.mark.parametrize("cmd,args,expected", [
    ("gen_cover", ["--generations", "50"], 50),
    ("gen_default", [], 2000),
])
def test_generations_flag(tmp_path, monkeypatch, cmd, args, expected):
    """
    B: 验证 --generations 参数能够覆盖或保持默认，并传递给 run()。
    - 用 monkeypatch 替换 retina.run，记录收到的 generations 并退出。
    """
    captured_init = {}
    captured_run = {}

    def fake_init(self, substrate, cppn, params):
        captured_init['use_3d'] = params.get("use_3d", False)

    def fake_run(generations=2000, use_3d=False):
        captured_run['generations'] = generations
        sys.exit(0)

    monkeypatch.setattr(ESNetwork, "__init__", fake_init)
    monkeypatch.setattr(retina, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["es_hyperneat_retina.py", *args])

    with pytest.raises(SystemExit) as exc:
        retina.main()
    assert exc.value.code == 0
    assert captured_init.get("use_3d", False) is False
    assert captured_run["generations"] == expected


def test_generations_invalid(tmp_path):
    """
    B-3: 非法值 --generations -1，应以非零退出码结束，stderr 包含提示信息。
    依赖 argparse 的类型校验。
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-m", "pureples.experiments.retina.es_hyperneat_retina", "--generations", "-1"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode != 0
    err = result.stderr.decode().lower()
    assert "value must be positive" in err or "invalid" in err


def test_combined_flags(tmp_path, monkeypatch):
    """
    C: 同时传 --use-3d 和 --generations 10，顺序无关。
    验证两者互不干扰，分别到达 ESNetwork.__init__ 和 run()。
    """
    captured_init = {}
    captured_run = {}

    def fake_init(self, substrate, cppn, params):
        captured_init['use_3d'] = params.get("use_3d", False)

    def fake_run(generations=2000, use_3d=False):
        captured_run['generations'] = generations
        sys.exit(0)

    monkeypatch.setattr(ESNetwork, "__init__", fake_init)
    monkeypatch.setattr(retina, "run", fake_run)

    for args in (["--use-3d", "--generations", "10"], ["--generations", "10", "--use-3d"]):
        captured_init.clear()
        captured_run.clear()
        monkeypatch.setattr(sys, "argv", ["es_hyperneat_retina.py", *args])

        with pytest.raises(SystemExit) as exc:
            retina.main()
        assert exc.value.code == 0
        assert captured_init.get("use_3d", False) is True
        assert captured_run.get("generations") == 10

