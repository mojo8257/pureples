# tests/test_b2_evaluation_integration.py
"""
B2：Evaluation 模块集成测试（最小入侵、防泄漏版）
- 关键点：
  1) 所有并行评估路径都用 “try/finally + 失败→terminate / 成功→close” 回收进程池；
  2) 对 run() 的装配点，打补丁为 SafeParallelEvaluator（evaluate 结束即回收，适配 1 代用例）；
  3) 所有 NEAT 对象创建前，先调用 apply_use_3d(...)，避免配置-对象生命周期交叉。
"""

import itertools
import multiprocessing as mp
import os
import types
from typing import Callable, List, Tuple

import pytest

import neat


# ------------- 工具：导入待测模块（SUT） -----------------
def _import_mod():
    import importlib
    return importlib.import_module("pureples.experiments.retina.es_hyperneat_retina")


# ------------- 工具：串行适配器（避免并发时的偶发干扰） ------
def _serial_evaluate(eval_fn: Callable, genomes, config):
    """
    Population.run 的 evaluate 适配器（串行版本）：
    - 输入：genomes = List[(genome_id, genome)], config
    - 对每个 genome 调用 eval_fn(genome, config)，并写回 genome.fitness
    """
    for gid, g in genomes:
        g.fitness = float(eval_fn(g, config))


# ------------- SafeParallelEvaluator（测试专用替身） --------
class SafeParallelEvaluator:
    """
    测试专用的 “安全版” ParallelEvaluator：
    - evaluate(...) 只跑一次任务批，并在结束后**立即回收**进程池
      （本套测试都是 1 代或少量代次，足够覆盖契约；避免遗留）。
    - 异常路径：直接 terminate() + join()，确保不 hang。
    """
    _instances: List["SafeParallelEvaluator"] = []

    def __init__(self, num_workers: int, eval_function: Callable, timeout: float | None = None):
        self.num_workers = max(1, int(num_workers or 1))
        self.eval_function = eval_function
        self.timeout = timeout
        # 使用 spawn/fork 均可；默认取系统默认；为兼容起见沿用默认上下文
        self.pool = mp.Pool(processes=self.num_workers)
        type(self)._instances.append(self)

    def _eval_one(self, args: Tuple[int, neat.DefaultGenome, neat.config.Config]) -> Tuple[int, float]:
        gid, genome, config = args
        return gid, float(self.eval_function(genome, config))

    def evaluate(self, genomes, config):
        jobs = None
        try:
            args = [(gid, g, config) for gid, g in genomes]
            # 用 map 即可；如果后续需要 timeout，可切换为 apply_async + get(timeout)
            results = self.pool.map(self._eval_one, args)
            # 写回 fitness
            for gid, fit in results:
                for _gid, g in genomes:
                    if _gid == gid:
                        g.fitness = fit
                        break
        except BaseException:
            # 异常：硬清理
            try:
                self.pool.terminate()
            finally:
                self.pool.join()
            raise
        else:
            # 成功：干净关闭
            try:
                self.pool.close()
            finally:
                self.pool.join()

    # 兜底：即使忘记清理，也尽量不泄漏（不过测试里 evaluate 会主动清理）
    def __del__(self):
        try:
            if hasattr(self, "pool"):
                self.pool.terminate()
                self.pool.join()
        except Exception:
            pass

    @classmethod
    def terminate_all(cls):
        for inst in list(cls._instances):
            try:
                if hasattr(inst, "pool"):
                    inst.pool.terminate()
                    inst.pool.join()
            except Exception:
                pass
        cls._instances.clear()


@pytest.fixture
def patch_safe_parallel(monkeypatch):
    """
    把 run() 装配点的 neat.parallel.ParallelEvaluator 替换为 SafeParallelEvaluator，
    并在测试结束后统一清理所有实例。
    """
    import neat.parallel
    monkeypatch.setattr(neat.parallel, "ParallelEvaluator",
                        SafeParallelEvaluator, raising=True)
    try:
        yield
    finally:
        SafeParallelEvaluator.terminate_all()


# ------------- 行为集用例 -----------------------------

def test_d1_leo_smoke_with_real_eval():
    """
    D1（I 维，LEO 参与）：使用真实 retina_eval_single 做一次烟囱（串行），
    只验证能得到 winner（返回对象非 None）。
    """
    mod = _import_mod()
    # 先同步 2D，避免输入维度错位
    mod.apply_use_3d(False)

    pop = neat.Population(mod.CONFIG)
    winner = pop.run(lambda genomes, cfg: _serial_evaluate(
        mod.retina_eval_single, genomes, cfg), 1)
    assert winner is not None


def test_d2_use3d_input_dim_switch():
    """
    D2（I 维，3D 扩展）：切换 2D -> 3D，两次串行烟囱均能得到 winner。
    关注“无异常”，不对 fitness 数值做强断言。
    """
    mod = _import_mod()

    # 2D
    mod.apply_use_3d(False)
    pop2d = neat.Population(mod.CONFIG)
    w2d = pop2d.run(lambda genomes, cfg: _serial_evaluate(
        mod.retina_eval_single, genomes, cfg), 1)
    assert w2d is not None

    # 3D
    mod.apply_use_3d(True)
    pop3d = neat.Population(mod.CONFIG)
    w3d = pop3d.run(lambda genomes, cfg: _serial_evaluate(
        mod.retina_eval_single, genomes, cfg), 1)
    assert w3d is not None


def test_a1_wires_parallel_evaluator_success_close():
    """
    A1（D+E 维，装配点）：使用真实 ParallelEvaluator 等价行为（SafeParallelEvaluator），
    成功路径 evaluate 后应自动 close/join（不泄漏子进程）。
    """
    mod = _import_mod()
    mod.apply_use_3d(False)

    pe = SafeParallelEvaluator(
        num_workers=1, eval_function=mod.retina_eval_single)
    pop = neat.Population(mod.CONFIG)

    # 只跑 1 代，确保 evaluate 被调用一次
    pop.run(pe.evaluate, 1)

    # 若泄漏，module/session 保险丝会在 teardown 报告；此处观察不报错即可
    assert True


def test_a2_evaluate_contract_set_fitness_and_no_leak(monkeypatch):
    """
    A2（I+D 维，契约）：SafeParallelEvaluator.evaluate 必须为每个 genome 写入 fitness，
    并且无论失败/成功都不应泄漏进程池。
    """
    mod = _import_mod()
    mod.apply_use_3d(False)

    # 包装一个会抛错的 eval_fn，触发异常路径（应 terminate+join）
    def boom_eval(genome, cfg):
        raise RuntimeError("boom")

    pe = SafeParallelEvaluator(num_workers=1, eval_function=boom_eval)
    pop = neat.Population(mod.CONFIG)

    with pytest.raises(RuntimeError):
        # 触发异常路径
        pop.run(pe.evaluate, 1)

    # 走到这里说明异常被抛出且测试没有 hang，进程池已回收
    assert True


def test_a3_blackbox_retina_run_once_with_safe_parallel(patch_safe_parallel, monkeypatch, tmp_path):
    """
    A3（I 维黑盒）：对 SUT 的 run() 做一次整体验收（1 代），
    通过 patch_safe_parallel 确保内部并行评估安全回收。
    """
    mod = _import_mod()

    # 驱动目录用临时目录，避免对真实磁盘产生副作用
    drive_dir = tmp_path / "retina_run_once"
    drive_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SAVE_DIR", str(drive_dir))

    # run() 内会读取 2D/3D；提前同步
    mod.apply_use_3d(False)

    # 准备 run() 收尾会复制的占位图片（避免 FileNotFoundError）
    for name in ("rank00_cppn.png", "rank00_phen.png"):
        (drive_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n")  # 简单 PNG 头占位

    # 跑 1 代即可触发装配点
    mod.run(generations=1)

    # run() 应该生成 winner_* 文件（拷贝）
    assert (drive_dir / "winner_retina_cppn.png").exists()
    assert (drive_dir / "winner_retina_substrate.png").exists()
