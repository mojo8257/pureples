"""
ES‑HyperNEAT Retina 实验脚本
--------------------------------
支持 neat‑python + PurePLES，在 Colab 环境下可直接运行。
本版本根据“方案 A” 重构了 2D/3D 开关的同步逻辑：
  • 单一函数 apply_use_3d(flag) 负责写入所有全局状态。
  • CLI 解析后才决定最终 flag，并写回 os.environ 确保 spawn 子进程一致。
  • 顶层在 import 时只读环境变量一次，以便子进程自动同步。
"""

from __future__ import annotations

import argparse
import importlib.resources as pkg_res
import itertools
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys
from typing import List

import neat
import neat.nn
from neat.reporting import BaseReporter

from pureples.shared.substrate import Substrate
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.visualize import draw_net

# ──────────────────────────────────────────────────────────────────────────────
# CLI 辅助
# -----------------------------------------------------------------------------

def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return ivalue

# ──────────────────────────────────────────────────────────────────────────────
# 0. 数据常量：合法的 2×2 图案（论文 Fig‑15）
# -----------------------------------------------------------------------------
VALID_PATTERNS = {
    (1, 1, 1, 1), (1, 1, 0, 0),
    (0, 0, 1, 1), (1, 0, 1, 0),
    (0, 1, 0, 1), (1, 0, 0, 0),
    (0, 1, 0, 0), (0, 0, 1, 0)
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. 几何坐标（Substrate）
# -----------------------------------------------------------------------------
INPUT_COORDS = [
    (-1.0, 0.0), (-0.33, 0.0), (0.33, 0.0), (1.0, 0.0),
    (-1.0, 0.0), (-0.33, 0.0), (0.33, 0.0), (1.0, 0.0),
]
OUTPUT_COORDS = [(-0.5, 1.0), (0.5, 1.0)]
SUBSTRATE = Substrate(INPUT_COORDS, OUTPUT_COORDS)

# ──────────────────────────────────────────────────────────────────────────────
# 2. ES‑HyperNEAT 专用参数（Appendix 1）
# -----------------------------------------------------------------------------

def es_params() -> dict:
    """默认的 ES‑HyperNEAT 参数表。后续会被 CONFIG 覆盖三项。"""
    return dict(
        initial_depth=2,
        max_depth=5,
        variance_threshold=0.03,
        band_threshold=0.3,
        iteration_level=1,
        division_threshold=0.5,
        max_weight=5.0,
        activation="sigmoid",
        enable_leo=True,
        leo_threshold=0.0,
        locality_seed="xaxis",
    )

ES_PARAMS = es_params()

# ──────────────────────────────────────────────────────────────────────────────
# 3. 读取 NEAT‑CPPN 配置文件
# -----------------------------------------------------------------------------
CFG_PATH = pkg_res.files(__package__).joinpath("config_cppn_retina")
CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    str(CFG_PATH),
)

# 把 LEO / locality‑seed 三项同步到 genome_config，并覆盖 ES_PARAMS
CONFIG.genome_config.enable_leo = CONFIG.enable_leo
CONFIG.genome_config.locality_seed = CONFIG.locality_seed
CONFIG.genome_config.leo_bias_default = CONFIG.leo_bias_default

if CONFIG.genome_config.enable_leo:
    CONFIG.genome_config.leo_output_key = CONFIG.genome_config.num_outputs
    if CONFIG.genome_config.leo_output_key not in CONFIG.genome_config.output_keys:
        CONFIG.genome_config.output_keys.append(CONFIG.genome_config.leo_output_key)

ES_PARAMS.update(
    enable_leo=CONFIG.enable_leo,
    leo_threshold=CONFIG.leo_threshold,
    locality_seed=CONFIG.locality_seed,
)

# ──────────────────────────────────────────────────────────────────────────────
# 3‑D 开关同步逻辑
# -----------------------------------------------------------------------------

def apply_use_3d(flag: bool) -> None:
    """同步 3‑D 开关到所有全局状态（ES_PARAMS / CONFIG）。"""
    ES_PARAMS["use_3d"] = flag
    CONFIG.genome_config.num_inputs = 7 if flag else 5
    CONFIG.genome_config.input_keys = list(range(-CONFIG.genome_config.num_inputs, 0))

# 顶层：spawn 子进程 import 时自动同步（环境变量兜底）
apply_use_3d(os.getenv("ES_USE_3D", "0") == "1")

# ──────────────────────────────────────────────────────────────────────────────
# 4. 评估函数 & 并行桩
# -----------------------------------------------------------------------------

def retina_eval_single(genome, neat_config):
    """单个 genome 的适应度评估（供 ParallelEvaluator 调用）。"""
    cppn = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    phen = ESNetwork(SUBSTRATE, cppn, ES_PARAMS).create_phenotype_network()
    error = 0.0
    for pattern in itertools.product((0, 1), repeat=8):
        left, right = pattern[:4], pattern[4:]
        tgt_left = 1.0 if left in VALID_PATTERNS else -1.0
        tgt_right = 1.0 if right in VALID_PATTERNS else -1.0
        inp = [3.0 if p else -3.0 for p in pattern]
        out_left, out_right = phen.activate(inp)
        error += (out_left - tgt_left) ** 2 + (out_right - tgt_right) ** 2
    return 1000.0 / (1.0 + error ** 2)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Reporter：每代保存前 K 名
# -----------------------------------------------------------------------------
class TopGenomeSaver(BaseReporter):
    def __init__(self, save_dir: str, top_k: int = 5):
        self.save_dir = save_dir
        self.top_k = top_k
        self.cur_gen = 0

    def start_generation(self, generation: int):
        self.cur_gen = generation

    def post_evaluate(self, config, population, species, best_genome):
        gen = self.cur_gen
        top = sorted(population.values(), key=lambda g: g.fitness or -1, reverse=True)[
            : self.top_k
        ]
        for rank, g in enumerate(top):
            tag = f"rank{rank:02d}"
            pkl = os.path.join(self.save_dir, f"{tag}.pkl")
            with open(pkl, "wb") as f:
                pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            draw_net(cppn, filename=os.path.join(self.save_dir, f"{tag}_cppn.png"))
            esnet = ESNetwork(SUBSTRATE, cppn, ES_PARAMS)
            esnet.create_phenotype_network(
                filename=os.path.join(self.save_dir, f"{tag}_phen.png")
            )
        with open(os.path.join(self.save_dir, "_latest_gen.txt"), "w") as f:
            f.write(str(gen))

# ──────────────────────────────────────────────────────────────────────────────
# 6. 主训练函数
# -----------------------------------------------------------------------------

def resume_from_checkpoint(pop_size: int):
    drive_dir = os.environ.get("SAVE_DIR", "/content/drive/MyDrive/ESHyperNEAT_Retina")
    ckpts = [f for f in os.listdir(drive_dir) if re.match(r"chkpt-\\d+", f)]
    if not ckpts:
        return None
    latest = max(ckpts, key=lambda f: int(f.split("-")[1]))
    print("[INFO] 恢复自 checkpoint →", latest)
    return neat.Checkpointer.restore_checkpoint(os.path.join(drive_dir, latest))


def run(generations: int = 2000):
    """核心训练循环。"""
    drive_dir = os.environ.get("SAVE_DIR", "/content/drive/MyDrive/ESHyperNEAT_Retina")
    os.makedirs(drive_dir, exist_ok=True)

    pop = resume_from_checkpoint(CONFIG.pop_size) or neat.Population(CONFIG)

    # Reporters
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(TopGenomeSaver(drive_dir, top_k=5))
    pop.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            time_interval_seconds=1800,
            filename_prefix=os.path.join(drive_dir, "chkpt-"),
        )
    )

    n_cpu = os.cpu_count() or 1
    pe = neat.parallel.ParallelEvaluator(n_cpu, retina_eval_single)
    winner = pop.run(pe.evaluate, generations)

    # 收尾：把最终结果复制为 winner_* 文件
    shutil.copy(os.path.join(drive_dir, "rank00_cppn.png"), os.path.join(drive_dir, "winner_retina_cppn.png"))
    shutil.copy(os.path.join(drive_dir, "rank00_phen.png"), os.path.join(drive_dir, "winner_retina_substrate.png"))

    print("\n=== Retina‑ES‑HyperNEAT 训练结束 ===")
    print("Winner Genome ID:", winner.key, " Fitness=", winner.fitness)

    cppn = neat.nn.FeedForwardNetwork.create(winner, CONFIG)
    final_esnet = ESNetwork(SUBSTRATE, cppn, ES_PARAMS)

    phen_png = os.path.join(drive_dir, "winner_retina_substrate.png")
    final_esnet.create_phenotype_network(filename=phen_png)
    cppn_png = os.path.join(drive_dir, "winner_retina_cppn.png")
    draw_net(cppn, filename=cppn_png)
    cppn_pkl = os.path.join(drive_dir, "winner_retina_cppn.pkl")
    with open(cppn_pkl, "wb") as f:
        pickle.dump(cppn, f, pickle.HIGHEST_PROTOCOL)

# ──────────────────────────────────────────────────────────────────────────────
# 7. CLI 入口
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run ES‑HyperNEAT retina experiment")
    parser.add_argument("--use-3d", action="store_true", help="Enable 3D phenotype generation")
    parser.add_argument("--generations", type=positive_int, default=2000, help="Number of generations to run")
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # CLI 高于环境：只看 CLI，缺省 False
    final_flag = args.use_3d
    apply_use_3d(final_flag)
    os.environ["ES_USE_3D"] = "1" if final_flag else "0"

    if args.dry_run:
        sys.exit(0)

    mp.set_start_method("fork", force=True)
    run(generations=args.generations)


if __name__ == "__main__":
    main()
