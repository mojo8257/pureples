# tests/test_a3_population_wiring.py
from __future__ import annotations

import os
import re
import importlib
import types
from functools import lru_cache
from typing import Any

import pytest


# --------- helpers: safe import (abs first, then relative fallback) ---------
@lru_cache()
def _import_retina_mod():
    try:
        return importlib.import_module(
            "pureples.experiments.retina.es_hyperneat_retina"
        )
    except (ModuleNotFoundError, ImportError):
        return importlib.import_module(
            "..experiments.retina.es_hyperneat_retina", package="pureples"
        )


# --------- wiring stubs (lightweight & side-effect free) ---------
class PopStub:
    """Minimal neat.Population replacement: records reporter registrations and run() calls."""

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.reporters: list[Any] = []
        self.run_calls: list[tuple[tuple, dict]] = []

    def add_reporter(self, reporter: Any) -> None:
        self.reporters.append(reporter)

    # neat.Population.run(eval_fn, generations)
    def run(self, *args, **kwargs) -> Any:
        self.run_calls.append((args, kwargs))
        return None


class CheckpointerStub:
    """Emulates neat.Checkpointer. Also provides a restore_checkpoint spy."""

    created: list["CheckpointerStub"] = []
    restore_called_paths: list[str] = []

    def __init__(
        self,
        generation_interval: int = 0,
        time_interval_seconds: int = 0,
        filename_prefix: str = "",
    ) -> None:
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds
        self.filename_prefix = filename_prefix
        CheckpointerStub.created.append(self)

    @classmethod
    def restore_checkpoint(cls, path: str) -> PopStub:
        cls.restore_called_paths.append(path)
        return PopStub()


class StdOutReporterStub:
    def __init__(self, verbose: bool) -> None:
        self.verbose = bool(verbose)


class StatisticsReporterStub:
    pass


class TopGenomeSaverStub:
    created: list["TopGenomeSaverStub"] = []

    def __init__(self, drive_dir: str, top_k: int = 5) -> None:
        self.drive_dir = drive_dir
        self.top_k = top_k
        TopGenomeSaverStub.created.append(self)


class ParallelEvaluatorStub:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def evaluate(self, *_a, **_k):
        return None


def _install_wiring_stubs(
    monkeypatch: pytest.MonkeyPatch,
    mod: types.ModuleType,
    *,
    resume_returns: PopStub | None,
    save_dir: str,
) -> dict[str, Any]:
    """
    Patch the retina module for wiring-only execution.
    """
    # SAVE_DIR environment
    monkeypatch.setenv("SAVE_DIR", save_dir)

    # Reset global collectors on stubs
    CheckpointerStub.created.clear()
    CheckpointerStub.restore_called_paths.clear()
    TopGenomeSaverStub.created.clear()

    # Patch neat submodules/classes that retina module uses
    neat = mod.neat

    # Population constructor returns PopStub (records reporters)
    monkeypatch.setattr(neat, "Population",
                        lambda cfg: PopStub(cfg), raising=True)

    # Checkpointer class (with .restore_checkpoint)
    monkeypatch.setattr(neat, "Checkpointer", CheckpointerStub, raising=True)

    # StdOutReporter and StatisticsReporter
    monkeypatch.setattr(neat.reporting, "StdOutReporter",
                        StdOutReporterStub, raising=True)
    monkeypatch.setattr(neat.statistics, "StatisticsReporter",
                        StatisticsReporterStub, raising=True)

    # TopGenomeSaver is defined in the retina experiment module
    monkeypatch.setattr(mod, "TopGenomeSaver",
                        TopGenomeSaverStub, raising=True)

    # If code builds a parallel evaluator, give a tiny stub
    if hasattr(neat, "parallel") and hasattr(neat.parallel, "ParallelEvaluator"):
        monkeypatch.setattr(neat.parallel, "ParallelEvaluator",
                            ParallelEvaluatorStub, raising=True)

    # Control resume branch deterministically
    calls: dict[str, Any] = {"resume_called": 0, "resume_args": None}

    def _resume_stub(pop_size: int):
        calls["resume_called"] += 1
        calls["resume_args"] = (pop_size,)
        return resume_returns

    monkeypatch.setattr(mod, "resume_from_checkpoint",
                        _resume_stub, raising=True)
    return calls


def _snapshot_invariants(mod: types.ModuleType) -> dict[str, Any]:
    gc = mod.CONFIG.genome_config
    es = mod.ES_PARAMS
    return {
        "num_inputs": int(gc.num_inputs),
        "input_keys": tuple(gc.input_keys),
        "use_3d": bool(es.get("use_3d")),
        "enable_leo": bool(es.get("enable_leo")),
        "leo_threshold": float(es.get("leo_threshold")),
        "locality_seed": es.get("locality_seed", None),
    }


def _touch(path: str | os.PathLike[str]) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "ab"):
        pass


def _ensure_rank_images(drive_dir: str | os.PathLike[str]) -> None:
    """Create both images that run() may try to copy at the end."""
    _touch(os.path.join(str(drive_dir), "rank00_cppn.png"))
    _touch(os.path.join(str(drive_dir), "rank00_phen.png"))


# ------------------------------ T1/T2/T3: resume_from_checkpoint ------------------------------

def _assert_latest_is(expected: int) -> None:
    assert CheckpointerStub.restore_called_paths, "restore_checkpoint was not called"
    m = re.search(r"chkpt-(\d+)", CheckpointerStub.restore_called_paths[-1])
    assert m, f"unexpected restore path: {CheckpointerStub.restore_called_paths[-1]!r}"
    assert int(m.group(
        1)) == expected, f"expected latest checkpoint '{expected}', got {m.group(1)}"


def test_T1_resume_selects_latest_checkpoint(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    """
    与真实实现保持一致：
    - 只识别严格的 'chkpt-\\d+'（无扩展名）；
    - 从环境变量 SAVE_DIR 读取目录；
    - 使用 neat.Checkpointer.restore_checkpoint 恢复。
    """
    mod = _import_retina_mod()

    drive_dir = tmp_path / "save"
    drive_dir.mkdir(parents=True, exist_ok=True)
    # 只创建无扩展名的候选，确保最新号为 10
    for name in ("chkpt-1", "chkpt-9", "chkpt-10"):
        (drive_dir / name).write_text("")

    # Patch env 和 Checkpointer（只监听/返回 PopStub）
    CheckpointerStub.created.clear()
    CheckpointerStub.restore_called_paths.clear()
    monkeypatch.setenv("SAVE_DIR", str(drive_dir))
    monkeypatch.setattr(mod.neat, "Checkpointer",
                        CheckpointerStub, raising=True)

    pop = mod.resume_from_checkpoint(pop_size=mod.CONFIG.pop_size)
    assert isinstance(pop, PopStub)
    _assert_latest_is(10)


def test_T2_resume_returns_none_when_no_ckpt(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    mod = _import_retina_mod()

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir(parents=True, exist_ok=True)

    CheckpointerStub.created.clear()
    CheckpointerStub.restore_called_paths.clear()
    monkeypatch.setenv("SAVE_DIR", str(empty_dir))
    monkeypatch.setattr(mod.neat, "Checkpointer",
                        CheckpointerStub, raising=True)

    pop = mod.resume_from_checkpoint(pop_size=mod.CONFIG.pop_size)
    assert pop is None
    assert CheckpointerStub.restore_called_paths == []


def test_T3_resume_ignores_noise_and_parses_numbers(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    mod = _import_retina_mod()

    drive_dir = tmp_path / "mix"
    drive_dir.mkdir(parents=True, exist_ok=True)
    # 既有合法也有噪声；真实实现只会匹配无扩展名的 'chkpt-\\d+'
    for name in ("chkpt-001", "chkpt-20", "chkpt-x", "checkpoint-3"):
        (drive_dir / name).write_text("")

    CheckpointerStub.created.clear()
    CheckpointerStub.restore_called_paths.clear()
    monkeypatch.setenv("SAVE_DIR", str(drive_dir))
    monkeypatch.setattr(mod.neat, "Checkpointer",
                        CheckpointerStub, raising=True)

    pop = mod.resume_from_checkpoint(pop_size=mod.CONFIG.pop_size)
    assert isinstance(pop, PopStub)
    _assert_latest_is(20)


# ------------------------------ T5/T6: run wiring on restored/new pop ------------------------------

def test_T5_run_wires_reporters_on_restored_pop(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    mod = _import_retina_mod()
    drive_dir = tmp_path / "wired_restore"
    drive_dir.mkdir(parents=True, exist_ok=True)

    calls = _install_wiring_stubs(
        monkeypatch, mod, resume_returns=PopStub(), save_dir=str(drive_dir))

    # 准备 run() 末尾会拷贝的两张图片
    _ensure_rank_images(str(drive_dir))

    # 触发接线；PopStub.run() 是 no-op
    mod.run(generations=3)

    # resume 被调用一次，参数为 CONFIG.pop_size
    assert calls["resume_called"] == 1
    assert isinstance(calls["resume_args"], tuple) and len(
        calls["resume_args"]) == 1
    assert calls["resume_args"][0] == mod.CONFIG.pop_size

    # 报表器/检查点接线各 1 次
    assert len(TopGenomeSaverStub.created) == 1
    assert len(CheckpointerStub.created) == 1


def test_T6_run_wires_reporters_on_new_pop(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    mod = _import_retina_mod()
    drive_dir = tmp_path / "wired_new"
    drive_dir.mkdir(parents=True, exist_ok=True)

    calls = _install_wiring_stubs(
        monkeypatch, mod, resume_returns=None, save_dir=str(drive_dir))

    _ensure_rank_images(str(drive_dir))

    mod.run(generations=2)

    # resume 调用但返回 None -> 走新建 Population 分支
    assert calls["resume_called"] == 1
    assert len(TopGenomeSaverStub.created) == 1
    assert len(CheckpointerStub.created) == 1


# ------------------------------ T7/T8: reporter parameter checks ------------------------------

def test_T7_checkpointer_params(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    mod = _import_retina_mod()
    drive_dir = tmp_path / "params_ckpt"
    drive_dir.mkdir(parents=True, exist_ok=True)

    _install_wiring_stubs(
        monkeypatch, mod, resume_returns=None, save_dir=str(drive_dir))

    _ensure_rank_images(str(drive_dir))

    mod.run(generations=1)

    assert len(CheckpointerStub.created) == 1
    ck = CheckpointerStub.created[0]
    # 与实现保持一致
    assert getattr(ck, "generation_interval") == 10
    assert getattr(ck, "time_interval_seconds") == 1800
    assert getattr(ck, "filename_prefix").startswith(
        os.path.join(str(drive_dir), "chkpt-"))


def test_T8_topgenomesaver_params(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    mod = _import_retina_mod()
    drive_dir = tmp_path / "params_topk"
    drive_dir.mkdir(parents=True, exist_ok=True)

    _install_wiring_stubs(
        monkeypatch, mod, resume_returns=None, save_dir=str(drive_dir))

    _ensure_rank_images(str(drive_dir))

    mod.run(generations=1)

    assert len(TopGenomeSaverStub.created) == 1
    tg = TopGenomeSaverStub.created[0]
    assert getattr(tg, "drive_dir") == str(drive_dir)
    assert getattr(tg, "top_k") == 5


# ------------------------------ T10/T11: invariants passthrough (A2 coherence) ------------------------------

@pytest.mark.parametrize("use3d", [False, True])
def test_T10_invariants_passthrough_on_restored_pop(use3d: bool, tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    mod = _import_retina_mod()
    drive_dir = tmp_path / "inv_restore"
    drive_dir.mkdir(parents=True, exist_ok=True)

    # 通过 A2 API 切换 2D/3D
    mod.apply_use_3d(use3d)
    before = _snapshot_invariants(mod)

    _install_wiring_stubs(
        monkeypatch, mod, resume_returns=PopStub(), save_dir=str(drive_dir))

    _ensure_rank_images(str(drive_dir))

    mod.run(generations=1)

    after = _snapshot_invariants(mod)
    assert after == before, "A3 不应覆盖 A2 的 CONFIG/ES_PARAMS"


@pytest.mark.parametrize("use3d", [False, True])
def test_T11_invariants_passthrough_on_new_pop(use3d: bool, tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    mod = _import_retina_mod()
    drive_dir = tmp_path / "inv_new"
    drive_dir.mkdir(parents=True, exist_ok=True)

    mod.apply_use_3d(use3d)
    before = _snapshot_invariants(mod)

    _install_wiring_stubs(
        monkeypatch, mod, resume_returns=None, save_dir=str(drive_dir))

    _ensure_rank_images(str(drive_dir))

    mod.run(generations=1)

    after = _snapshot_invariants(mod)
    assert after == before, "A3 不应覆盖 A2 的 CONFIG/ES_PARAMS"


# ------------------------------ T12: prefix/dir consistency ------------------------------

def test_T12_prefix_and_dir_consistency(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    mod = _import_retina_mod()
    drive_dir = tmp_path / "consistency"
    drive_dir.mkdir(parents=True, exist_ok=True)

    _install_wiring_stubs(
        monkeypatch, mod, resume_returns=None, save_dir=str(drive_dir))

    _ensure_rank_images(str(drive_dir))

    mod.run(generations=1)

    assert len(CheckpointerStub.created) == 1
    assert len(TopGenomeSaverStub.created) == 1

    ck = CheckpointerStub.created[0]
    tg = TopGenomeSaverStub.created[0]
    # Checkpointer 写入目录与 TopGenomeSaver 目标目录一致
    ck_dir = os.path.dirname(getattr(ck, "filename_prefix"))
    assert ck_dir == tg.drive_dir == str(drive_dir)


# ------------------------------ A1: blackbox I-only acceptance ------------------------------

def test_A1_blackbox_minimal_wiring(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch):
    """
    I 维黑盒接线验收：仅校验“可返回已接线的 pop（间接通过 stub 观察）”。
    不断言演化行为，不落盘真实 checkpoint。
    """
    mod = _import_retina_mod()
    drive_dir = tmp_path / "blackbox"
    drive_dir.mkdir(parents=True, exist_ok=True)

    _install_wiring_stubs(
        monkeypatch, mod, resume_returns=None, save_dir=str(drive_dir))

    _ensure_rank_images(str(drive_dir))

    # 运行不应抛错，接线通过 Stub 被观测
    mod.run(generations=1)

    assert len(CheckpointerStub.created) == 1
    assert len(TopGenomeSaverStub.created) == 1
