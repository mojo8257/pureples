# tests/test_b1_evolution_integration.py
# -*- coding: utf-8 -*-

import os
import io
import pathlib
import pickle
import typing as t

import pytest
import neat

# 被测上层（含 CONFIG / ES_PARAMS / apply_use_3d / TopGenomeSaver / retina_eval_single）
from pureples.experiments.retina import es_hyperneat_retina as mod
from neat.reporting import BaseReporter


# ──────────────────────────────────────────────────────────────────────────────
# 工具与桩
# -----------------------------------------------------------------------------


class SpyReporter(BaseReporter):
    """记录调用序列与关键参数的 Reporter。"""

    def __init__(self):
        self.calls: t.List[str] = []
        self.start_gens: t.List[int] = []
        self.post_gens: t.List[int] = []
        self.post_best_keys: t.List[int] = []
        self.end_gens: t.List[int] = []

    def start_generation(self, generation):
        self.calls.append("start")
        self.start_gens.append(generation)

    def post_evaluate(self, config, population, species, best_genome):
        self.calls.append("post")
        # 记录当代 best 的 key（若有）
        if best_genome is not None:
            self.post_best_keys.append(getattr(best_genome, "key", None))
        else:
            self.post_best_keys.append(None)
        self.post_gens.append(species.generation if hasattr(
            species, "generation") else None)

    def end_generation(self, config, population, species):
        self.calls.append("end")
        self.end_gens.append(species.generation if hasattr(
            species, "generation") else None)


def _fitness_const_and_assert_inputs(expected_num_inputs: int):
    """返回一个评估桩：断言输入维度并把所有个体 fitness 置为 1.0。"""

    def _fn(genomes, config):
        assert config.genome_config.num_inputs == expected_num_inputs
        for _, g in genomes:
            g.fitness = 1.0

    return _fn


def _fitness_rank_by_key_desc():
    """按 key 升序/降序设定分数，便于确定性地选出 winner。"""

    def _fn(genomes, _config):
        # 令 key 越大分数越高（确定性 winner）
        for k, g in genomes:
            g.fitness = float(k)

    return _fn


class _ESNetNoop:
    """给 TopGenomeSaver 用的轻量 ESNetwork 替身（只提供 create_phenotype_network）。"""

    def __init__(self, *_args, **_kwargs):
        pass

    def create_phenotype_network(self, *_, **__):
        # 返回一个带有 activate 的轻量对象，以免调用方依赖
        class _Phen:
            def activate(self, *_a, **_k):
                return (0.0, 0.0)

        return _Phen()


# ──────────────────────────────────────────────────────────────────────────────
# D3：Config-mapping 不变式（LEO / locality-seed）
# -----------------------------------------------------------------------------


def test_d3_leo_mapping_invariants():
    # 任意时候三项需保持一致
    assert mod.ES_PARAMS["enable_leo"] == mod.CONFIG.enable_leo
    assert mod.ES_PARAMS["leo_threshold"] == mod.CONFIG.leo_threshold

    if mod.CONFIG.genome_config.enable_leo:
        # 当启用 LEO 时，输出位与 output_keys 的关系需满足下述不变式
        assert mod.CONFIG.genome_config.leo_output_key == mod.CONFIG.genome_config.num_outputs
        assert mod.CONFIG.genome_config.leo_output_key in mod.CONFIG.genome_config.output_keys


def test_d3_locality_seed_invariants_with_3d_toggle():
    seed0 = mod.CONFIG.locality_seed
    mod.apply_use_3d(False)
    assert mod.ES_PARAMS["locality_seed"] == seed0
    mod.apply_use_3d(True)
    assert mod.ES_PARAMS["locality_seed"] == seed0


# ──────────────────────────────────────────────────────────────────────────────
# D1：评估器 × 2D/3D 维度一致性 & LEO 烟囱
# -----------------------------------------------------------------------------


def test_d1_inputs_2d_num_inputs_and_winner_non_null():
    mod.apply_use_3d(False)  # 2D
    pop = neat.Population(mod.CONFIG)
    spy = SpyReporter()
    pop.add_reporter(spy)

    winner = pop.run(_fitness_const_and_assert_inputs(
        expected_num_inputs=5), 1)
    assert winner is not None
    assert spy.calls == ["start", "post", "end"]


def test_d1_inputs_3d_num_inputs_and_winner_non_null():
    mod.apply_use_3d(True)  # 3D
    pop = neat.Population(mod.CONFIG)
    spy = SpyReporter()
    pop.add_reporter(spy)

    winner = pop.run(_fitness_const_and_assert_inputs(
        expected_num_inputs=7), 1)
    assert winner is not None
    assert spy.calls == ["start", "post", "end"]


def test_d1_leo_smoke_with_real_eval(monkeypatch):
    mod.apply_use_3d(False)
    pop = neat.Population(mod.CONFIG)
    # 使用 ParallelEvaluator 适配 single-genome 评估函数（并发度 1，避免多进程开销）
    pe = neat.parallel.ParallelEvaluator(1, mod.retina_eval_single)
    winner = pop.run(pe.evaluate, 1)
    assert winner is not None


# ──────────────────────────────────────────────────────────────────────────────
# D2：Reporters 行为（顺序/多代/落盘最小化）
# -----------------------------------------------------------------------------


def test_d2_reporter_call_sequence_and_best(monkeypatch):
    mod.apply_use_3d(False)
    pop = neat.Population(mod.CONFIG)

    # 预期当代 best（按 key 最大）
    expected_best_key = max(pop.population.keys())

    spy = SpyReporter()
    pop.add_reporter(spy)

    winner = pop.run(_fitness_rank_by_key_desc(), 1)
    assert winner is not None

    # 序列严格：start→post→end（各 1 次）
    assert spy.calls == ["start", "post", "end"]
    # post 阶段的 best_key 应与我们设定的一致
    assert spy.post_best_keys == [expected_best_key]


def test_d2_topk_minimal_disk_and_latest_gen(tmp_path, monkeypatch):
    """最小化落盘：top_k=2，仅验证 pkl 与 _latest_gen.txt。"""
    save_dir = tmp_path / "save"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 让 TopGenomeSaver 的绘图/表型生成变成 no-op，避免产生重 IO。
    monkeypatch.setattr(mod, "draw_net", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(mod, "ESNetwork", _ESNetNoop, raising=True)

    mod.apply_use_3d(False)
    pop = neat.Population(mod.CONFIG)

    topk = mod.TopGenomeSaver(str(save_dir), top_k=2)
    pop.add_reporter(topk)

    # 再挂一个 Spy 便于核对调用次数
    spy = SpyReporter()
    pop.add_reporter(spy)

    winner = pop.run(_fitness_rank_by_key_desc(), 2)
    assert winner is not None

    # Reporter 两代均被触发
    assert spy.calls == ["start", "post", "end", "start", "post", "end"]

    # _latest_gen.txt 记录末代编号（从 0 开始，n=2 时应为 1）
    latest_file = save_dir / "_latest_gen.txt"
    assert latest_file.exists()
    assert latest_file.read_text().strip() == "1"

    # 只校验 pkl（不要求 png）
    expect = {save_dir / "rank00.pkl", save_dir / "rank01.pkl"}
    found = set(p for p in save_dir.glob("rank*.pkl"))
    assert expect.issubset(found)


# ──────────────────────────────────────────────────────────────────────────────
# D4：繁殖/物种/停滞（契约）——代推进与规模不变式
# -----------------------------------------------------------------------------


def test_d4_generation_progress_and_population_size_invariant():
    mod.apply_use_3d(False)
    pop = neat.Population(mod.CONFIG)

    gen0 = pop.generation
    size0 = len(pop.population)

    winner = pop.run(_fitness_rank_by_key_desc(), 2)
    assert winner is not None

    # 代数推进 2
    assert pop.generation == gen0 + 2
    # 规模维持不变（NEAT 的常见不变式）
    assert len(pop.population) == size0


# （可选）负例：评估未写入 fitness 的容错 —— 若成本考虑可注释
@pytest.mark.optionalhook
def test_d4_negative_no_fitness_written_raises():
    mod.apply_use_3d(False)
    pop = neat.Population(mod.CONFIG)

    def _bad_eval(genomes, _config):
        # 故意不写入任何 fitness
        return

    with pytest.raises(Exception):
        pop.run(_bad_eval, 1)
