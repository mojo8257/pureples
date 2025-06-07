# ─── File: pureples/experiments/retina/es_hyperneat_retina.py ─────────────────

"""
ES-HyperNEAT Retina 实验脚本  ——  支持 Checkpointer + 写入 Google Drive
（等同于论文 Risi & Stanley 2012 中对 Retina 域的设置，
 已开启 LEO + x-axis 局部种子。）

依赖：
  - pureples    （已在 /content/pureples 安装）
  - neat-python （pip install neat-python）

请先在 Colab 中执行挂载 Drive、安装依赖等操作，然后再运行此脚本。
"""

import os
import pickle, shutil
import itertools
import neat
import neat.nn
from neat.reporting import BaseReporter

# 导入 pureples 内部需要的模块
from pureples.shared.substrate import Substrate
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.visualize import draw_net

import re

import multiprocessing as mp
from neat.parallel import ParallelEvaluator

# ──────────────────────────────────────────────────────────────────────────────
# 0. 全局常量：手动硬编码 “合法的 2×2 图案” (如论文 Fig.15 所示)
#    左右 Retina 要求相同的 8 个合法子模式
VALID_PATTERNS = {
    (1, 1, 1, 1), (1, 1, 0, 0),
    (0, 0, 1, 1), (1, 0, 1, 0),
    (0, 1, 0, 1), (1, 0, 0, 0),
    (0, 1, 0, 0), (0, 0, 1, 0)
}

# ──────────────────────────────────────────────────────────────────────────────
# 1.  几何坐标：8 个输入 + 2 个输出
#    - x ∈ { -1.0, -0.33, 0.33, 1.0 }   (左右对称)
#    - y = 0.0 表示 “输入层”， y = 1.0 表示 “输出层”
INPUT_COORDS = [
    (-1.0, 0.0), (-0.33, 0.0), (0.33, 0.0), (1.0, 0.0),
    (-1.0, 0.0), (-0.33, 0.0), (0.33, 0.0), (1.0, 0.0)
]
OUTPUT_COORDS = [
    (-0.5, 1.0),  # “左” 模块判别输出
    ( 0.5, 1.0)   # “右” 模块判别输出
]
SUBSTRATE = Substrate(INPUT_COORDS, OUTPUT_COORDS)

# ──────────────────────────────────────────────────────────────────────────────
# 2. ES-HyperNEAT 特殊参数（对应论文中 Appendix 1）
def es_params():
    """
    返回一个 dict，包含 ES-HyperNEAT 的参数。后面会动态从 CONFIG 里更新 enable_leo、leo_threshold、locality_seed。
    """
    return dict(
        initial_depth=2,         # 初始 quadtree 分辨率: 4×4
        max_depth=5,             # 最大 quadtree 分辨率: 32×32
        variance_threshold=0.03, # 变异阈值
        band_threshold=0.3,      # band 剪枝阈值
        iteration_level=1,       # 只迭代到隐藏神经元一层
        division_threshold=0.5,  # quadtree 分割阈值（论文中取值 0.5）
        max_weight=5.0,          # 最大权重映射
        activation="sigmoid",    # CPPN 内部使用 sigmoid 函数
        enable_leo=True,         # LEO 开关：后续由配置动态覆盖
        leo_threshold=0.0,       # LEO 阈值：后续由配置动态覆盖
        locality_seed="xaxis"    # x 轴局部种子
    )

ES_PARAMS = es_params()


# ──────────────────────────────────────────────────────────────────────────────
# 3. NEAT-CPPN 的配置文件（请确保相对路径正确，位于：pureples/experiments/retina）
CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    'pureples/experiments/retina/config_cppn_retina'
)

# 读取配置中的 LEO、阈值、局部种子三项，覆盖 ES_PARAMS
ES_PARAMS.update(dict(
    enable_leo   = CONFIG.enable_leo,
    leo_threshold= CONFIG.leo_threshold,
    locality_seed= CONFIG.locality_seed
))


# 保存冠军 + 前 4 名，可视化 CPPN/Phenotype
class TopGenomeSaver(BaseReporter):
    def __init__(self, save_dir, top_k=5):
        self.save_dir = save_dir
        self.top_k = top_k
        self.cur_gen = 0

    def start_generation(self, generation):
        self.cur_gen = generation

    # 在每一代评估完后被 NEAT 调用
    def post_evaluate(self, config, population, species, best_genome):
        gen = self.cur_gen
        # 1) 选出按 fitness 降序的前 k 个体
        top = sorted(population.values(), key=lambda g: g.fitness or -1, reverse=True)[:self.top_k]
        for rank, g in enumerate(top):
            tag = f'rank{rank:02d}'  # rank00 是冠军
            # —— 保存 genome 二进制 ——
            pkl = os.path.join(self.save_dir, f'{tag}.pkl')
            with open(pkl, 'wb') as f:
                pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
            # —— 画 CPPN  ——
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            draw_net(cppn, filename=os.path.join(self.save_dir, f'{tag}_cppn.png'))
            # —— 画 phenotype 网络 ——
            esnet = ESNetwork(SUBSTRATE, cppn, ES_PARAMS)
            esnet.create_phenotype_network(filename=os.path.join(self.save_dir, f'{tag}_phen.png'))
        # 2) 记录当前 generation 号，便于外部脚本识别是否刷新成功
        with open(os.path.join(self.save_dir, '_latest_gen.txt'), 'w') as f:
            f.write(str(gen))


# ──────────────────────────────────────────────────────────────────────────────
# 4. 评价函数  —— 对 256 种输入枚举，计算 MSE → 转换成 fitness
def retina_fitness(genomes, neat_config):
    """
    genomes: list of (genome_id, genome_obj) 元组
    neat_config: 从上面 CONFIG 里传入的配置
    """
    for _genome_id, genome in genomes:
        # 1) 用 neat-python 的 FeedForwardNetwork 把 genome（CPPN）解码
        cppn = neat.nn.FeedForwardNetwork.create(genome, neat_config)

        # 2) 用 ESNetwork 生成“Substrate 对应的前馈网络”
        es_net = ESNetwork(SUBSTRATE, cppn, ES_PARAMS)
        phen_net = es_net.create_phenotype_network()

        # 3) 遍历 2^8 = 256 种输入模式，累加平方误差
        error = 0.0
        for pattern in itertools.product((0, 1), repeat=8):
            # 左右图像各自 4 个像素
            left, right = pattern[:4], pattern[4:]

            # 目标输出：属于合法模式 => +1.0；否则 => -1.0
            target_left  =  1.0 if left  in VALID_PATTERNS else -1.0
            target_right =  1.0 if right in VALID_PATTERNS else -1.0

            # 输入数值映射：模式中 p=1 => +3.0，p=0 => -3.0
            # （参见论文 Fig-17 中输入映射区间大于 [-1,1]，以扩大差异）
            inp = [3.0 if p else -3.0 for p in pattern]

            out_left, out_right = phen_net.activate(inp)
            error += (out_left  - target_left )**2 + (out_right - target_right)**2

        # 4) 把平方误差转换成 fitness：f = 1000 / (1 + error^2)
        genome.fitness = 1000.0 / (1.0 + error**2)


def retina_eval_single(genome, neat_config):
    """供 ParallelEvaluator 调用，评估并返回一个 genome 的适应度"""
    # --- 与 retina_fitness() 内部的单个循环完全一致 ----------------
    cppn = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    phen = ESNetwork(SUBSTRATE, cppn, ES_PARAMS).create_phenotype_network()
    error = 0.0
    for pattern in itertools.product((0, 1), repeat=8):
        left, right = pattern[:4], pattern[4:]
        target_left = 1.0 if left in VALID_PATTERNS else -1.0
        target_right = 1.0 if right in VALID_PATTERNS else -1.0
        inp = [3.0 if p else -3.0 for p in pattern]
        out_left, out_right = phen.activate(inp)
        error += (out_left - target_left) ** 2 + (out_right - target_right) ** 2
    return 1000.0 / (1.0 + error ** 2)


def resume_from_checkpoint(pop_size):
    """如 SAVE_DIR 下已有 chkpt-*.pkl，则恢复最新；否则返回 None"""
    DRIVE_SAVE_DIR = os.environ.get('SAVE_DIR',
                                    '/content/drive/MyDrive/ESHyperNEAT_Retina')
    ckpts = [f for f in os.listdir(DRIVE_SAVE_DIR) if re.match(r'chkpt-\d+', f)]
    if not ckpts:
        return None

    latest = max(ckpts, key=lambda f: int(f.split('-')[1]))
    print("[INFO] 检测到现有 checkpoint →", latest, "，将从此处继续")
    return neat.Checkpointer.restore_checkpoint(
        os.path.join(DRIVE_SAVE_DIR, latest)
    )

# ──────────────────────────────────────────────────────────────────────────────
# 5. 主训练入口：带 Checkpointer，把中间结果写入 SAVE_DIR
def run(generations=2000):
    """
    主训练函数：
      - 把所有 Checkpointer 文件写到 SAVE_DIR
      - 训练完成后把 winner CPPN、phenotype net、可视化结果存到 SAVE_DIR
    """
    # 从环境变量里读 SAVE_DIR；如果没有，就用默认的相对路径
    DRIVE_SAVE_DIR = os.environ.get('SAVE_DIR', '/content/drive/MyDrive/ESHyperNEAT_Retina')
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

    # 1) 创建 Population 对象
    pop = resume_from_checkpoint(CONFIG.pop_size) or neat.Population(CONFIG)

    # 2) 添加必要的 Reporter
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)

    # 标准的 StdOutReporter，每隔一代打印一次（会刷新到 nohup 日志中）
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    # ★ TopK：每代刷新 champion+前 4 名
    pop.add_reporter(TopGenomeSaver(DRIVE_SAVE_DIR, top_k=5))

    # 添加 Checkpointer：每 10 代保存一次、或每 30 分钟保存一次，
    # filename_prefix 定位到 DRIVE_SAVE_DIR
    checkpointer = neat.Checkpointer(
        generation_interval=10,
        time_interval_seconds=1800,
        filename_prefix=os.path.join(DRIVE_SAVE_DIR, 'chkpt-')
    )
    pop.add_reporter(checkpointer)

    # 3) 构造 ParallelEvaluator  (自动 fork num_workers 个子进程)
    n_cpu = os.cpu_count() or 1
    pe = ParallelEvaluator(n_cpu, retina_eval_single)
    # 4) 正式开始跑（并行评估）
    winner = pop.run(pe.evaluate, generations)

    # run() 最终保存 Winner 之前，先清掉旧图（保证与“每代输出”一致）
    shutil.copy(os.path.join(DRIVE_SAVE_DIR, 'rank00_cppn.png'),
                os.path.join(DRIVE_SAVE_DIR, 'winner_retina_cppn.png'))
    shutil.copy(os.path.join(DRIVE_SAVE_DIR, 'rank00_phen.png'),
                os.path.join(DRIVE_SAVE_DIR, 'winner_retina_substrate.png'))

    # 4) 训练结束后的打印提示
    print("\n=== Retina-ES-HyperNEAT 训练结束 ===")
    print("Winner Genome ID:", winner.key, " Fitness=", winner.fitness)

    # ──────────────────────────────────────────────────────────────────────────
    # 5) 保存并可视化最终的 Winner CPPN  + 生成 phenotype net
    # （依赖于 neat.nn.FeedForwardNetwork 和 ESNetwork）
    cppn = neat.nn.FeedForwardNetwork.create(winner, CONFIG)
    final_esnet = ESNetwork(SUBSTRATE, cppn, ES_PARAMS)

    # a) 画出 phenotype 网络并保存到 Drive
    phen_png = os.path.join(DRIVE_SAVE_DIR, 'winner_retina_substrate.png')
    final_esnet.create_phenotype_network(filename=phen_png)
    print("Phenotype (Substrate) 图已保存至:", phen_png)

    # b) 画出 CPPN 网络并保存到 Drive
    cppn_png = os.path.join(DRIVE_SAVE_DIR, 'winner_retina_cppn.png')
    draw_net(cppn, filename=cppn_png)
    print("CPPN 结构图已保存至:", cppn_png)

    # c) 把 CPPN 对象 pickle 保存到 Drive
    cppn_pkl = os.path.join(DRIVE_SAVE_DIR, 'winner_retina_cppn.pkl')
    with open(cppn_pkl, 'wb') as f:
        pickle.dump(cppn, f, pickle.HIGHEST_PROTOCOL)
    print("CPPN pickle 已保存至:", cppn_pkl)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Colab/Linux 容器推荐显式设为 'spawn'，防止潜在 fork 问题
    mp.set_start_method('spawn', force=True)
    run()

