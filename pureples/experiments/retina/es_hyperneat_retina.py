"""
ES-HyperNEAT Retina experiment  --  LEΟ + x-locality seed  (Risi & Stanley 2012)

依赖:
  pureples   (本仓库已有)
  neat-python

放置路径:
  pureples/experiments/retina/retina_es_hyperneat.py
"""
import os
import sys
import pickle
import datetime
import itertools
import pickle
import neat
import neat.nn

from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork     # :contentReference[oaicite:0]{index=0}

# 1. 确保把 pureples 目录加到 Python 搜索路径中（假设你已经拉取到 /content/pureples）
sys.path.append('/content/pureples')
os.environ['PYTHONPATH'] = '/content/pureples'

# 2. 挂载 Google Drive，并创建一个专门的存储目录
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 在你的 Drive 根目录下创建一个文件夹用来保存 ES-HyperNEAT Retina 的结果
DRIVE_ROOT = '/content/drive/MyDrive'
SAVE_DIR    = os.path.join(DRIVE_ROOT, 'Retina_runs')
os.makedirs(SAVE_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# 0. 任务数据 ─ 8+8 合法 2×2 图案（硬编码自 Fig-15）
#    1 = ON, 0 = OFF  —— 顺序: 上左 UL, 上右 UR, 下左 LL, 下右 LR
VALID_PATTERNS = {
    (1, 1, 1, 1),     # 全亮
    (1, 1, 0, 0),     # 上亮
    (0, 0, 1, 1),     # 下亮
    (1, 0, 1, 0),     # 左列亮
    (0, 1, 0, 1),     # 右列亮
    (1, 0, 0, 0),     # 左上亮
    (0, 1, 0, 0),     # 右上亮
    (0, 0, 1, 0)      # 左下亮
}
# 两侧 retina 拥有同一模式集合；故左右合法集相同。

# ────────────────────────────────────────────────────────────────
# 1.  几何坐标
#    x = −1, −0.33, 0.33, 1   （左右对称）
#    y = 0.0  (输入) / 1.0 (输出)
INPUT_COORDS  = [(-1.0, 0.0), (-0.33, 0.0), (0.33, 0.0), (1.0, 0.0),
                 (-1.0, 0.0), (-0.33, 0.0), (0.33, 0.0), (1.0, 0.0)]   # 8 pixels
OUTPUT_COORDS = [(-0.5, 1.0), (0.5, 1.0)]                             # 左/右判别
SUBSTRATE     = Substrate(INPUT_COORDS, OUTPUT_COORDS)

# ────────────────────────────────────────────────────────────────
# 2. ES-HyperNEAT 专属参数（论文 Appendix 1）
def es_params():
    return dict(initial_depth=2,         # 4×4 初始采样
                max_depth=5,            # 32×32  (=2⁵)
                variance_threshold=0.03,
                band_threshold=0.3,
                iteration_level=1,
                division_threshold=0.5, # 取论文较大值
                max_weight=5.0,
                activation="sigmoid",
                enable_leo=True,        # --- 关键开关
                leo_threshold=0.0)

ES_PARAMS = es_params()                 # 动态更新见下

# ────────────────────────────────────────────────────────────────
# 3. CPPN-NEAT 配置
CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    'pureples/experiments/retina/config_cppn_retina'
)
# 读取 enable_leo / threshold / x-axis 种子标记
ES_PARAMS.update(dict(enable_leo   = CONFIG.enable_leo,
                      leo_threshold= CONFIG.leo_threshold))

# ────────────────────────────────────────────────────────────────
# 4. 评价函数
def retina_fitness(genomes, neat_config):
    for _, genome in genomes:
        cppn = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        net  = ESNetwork(SUBSTRATE, cppn, ES_PARAMS).create_phenotype_network()
        error = 0.0
        # 枚举 256 输入模式
        for pattern in itertools.product((0, 1), repeat=8):
            left, right = pattern[:4], pattern[4:]
            target_left  =  1.0 if left  in VALID_PATTERNS else -1.0
            target_right =  1.0 if right in VALID_PATTERNS else -1.0
            inp = [ 3.0 if p else -3.0 for p in pattern ]   # 幅值映射  [Fig-17] :contentReference[oaicite:1]{index=1}
            out_left, out_right = net.activate(inp)
            error += (out_left - target_left) ** 2 + (out_right - target_right) ** 2
        genome.fitness = 1000.0 / (1.0 + error ** 2)       # 公式 (§8.3) :contentReference[oaicite:2]{index=2}

# ────────────────────────────────────────────────────────────────
# 5. 主循环
def run(generations=2000):
    """
    运行 ES-HyperNEAT Retina 实验，并将 Checkpointer、winner CPPN/ANN、可视化都保存到
    SAVE_DIR 对应的 Google Drive 文件夹下。无需手动修改路径，脚本把所有结果都写到 Drive。
    """
    # 5.1 创建 Population 对象
    pop = neat.population.Population(CONFIG)

    # 5.2 追加 StatisticsReporter 和标准输出 reporter
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    # 5.3 追加 Checkpointer，每 10 代 或 每 30 分钟 存一次
    #     prefix 指向 Drive 下的 SAVE_DIR
    chk_prefix = os.path.join(SAVE_DIR, 'chkpt-')
    checkpointer = neat.Checkpointer(
        generation_interval=10,
        time_interval_seconds=1800,
        filename_prefix=chk_prefix
    )
    pop.add_reporter(checkpointer)

    # 5.4 运行 NEAT
    winner = pop.run(retina_fitness, generations)
    print("\n===== Retina-ES-HyperNEAT 完成 =====")
    print("Best genome key:", winner.key, "  fitness:", winner.fitness)

    # 5.5 所有输出都写到 Drive 下的 SAVE_DIR
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # 5.5.1 保存最终 winner CPPN 到 Drive
    cppn = neat.nn.FeedForwardNetwork.create(winner, CONFIG)
    final_cppn_path = os.path.join(SAVE_DIR, f'winner_cppn_{timestamp}.pkl')
    with open(final_cppn_path, 'wb') as f:
        pickle.dump(cppn, f, pickle.HIGHEST_PROTOCOL)
    print("Saved final CPPN to:", final_cppn_path)

    # 5.5.2 可视化 CPPN（保存 .png + .dot）到 Drive
    # 注意：draw_net 会自己生成 .png 与 .dot，路径不带扩展名即可。
    cppn_visual_path = os.path.join(SAVE_DIR, f'winner_cppn_vis_{timestamp}')
    draw_net(cppn, filename=cppn_visual_path)
    print("Saved CPPN visualization under:", cppn_visual_path + ".png")

    # 5.5.3 生成最终 ANN 结构图，并保存到 Drive
    #       draw_es 接受的 filename 同样不带扩展名，会写 .png
    from pureples.es_hyperneat.es_hyperneat import ESNetwork
    winner_net = ESNetwork(SUBSTRATE, cppn, ES_PARAMS) \
                    .create_phenotype_network(filename=os.path.join(SAVE_DIR, f'winner_ann_{timestamp}.png'))
    print("Saved final ANN structure image to Drive.")

    # 5.5.4 将当前种群统计数据（.pkl）也保存一份，以便离线分析
    stats_path = os.path.join(SAVE_DIR, f'stats_{timestamp}.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print("Saved StatisticsReporter data to:", stats_path)

    print("\n所有文件已备份到 Google Drive。")

if __name__ == '__main__':
    run()
