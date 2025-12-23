# EA_Operators/TuRBO.py
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TuRBOSubspace:
    """
    TuRBO信任区域 (Trust Region) 管理类
    对应论文公式 (5): Tk = (ck, Lk, Sk, Fk)
    """

    def __init__(self, dim, center_embedding, length_init=0.8, length_min=0.5 ** 7, length_max=1.6,
                 success_threshold=3, fail_threshold=20):
        self.dim = dim
        self.center = center_embedding  # ck: 区域中心 (语义向量)
        self.length = length_init  # Lk: 区域长度
        self.length_min = length_min
        self.length_max = length_max

        # 计数器
        self.success_count = 0  # Sk
        self.failure_count = 0  # Fk
        self.success_threshold = success_threshold  # tau_s
        self.fail_threshold = fail_threshold  # tau_f

    def sample_vector(self):
        """
        连续空间采样 (Continuous Space Sampling)
        对应论文公式 (8): z ~ N(ck, Lk * Id)
        """
        # 生成标准正态分布噪声
        noise = np.random.randn(self.dim)
        # 应用信任区域尺度和中心偏移
        z = self.center + (self.length * noise)
        return z

    def update_state(self, is_success):
        """
        根据HV增益更新区域状态
        对应论文公式 (7) 及伪代码 Step 3
        """
        if is_success:
            self.success_count += 1
            self.failure_count = 0
        else:
            self.success_count = 0
            self.failure_count += 1

        # 动态调整区域大小
        if self.success_count >= self.success_threshold:
            # 扩张区域: Lk <- min(2Lk, Lmax)
            self.length = min(2.0 * self.length, self.length_max)
            self.success_count = 0
            logger.info(f"Trust Region Expanded! Length: {self.length}")

        elif self.failure_count >= self.fail_threshold:
            # 收缩区域: Lk <- max(Lk/2, Lmin)
            self.length = max(self.length / 2.0, self.length_min)
            self.failure_count = 0
            logger.info(f"Trust Region Shrank. Length: {self.length}")

    def should_restart(self):
        """判断是否需要重启区域 [cite: 177]"""
        return self.length <= self.length_min


class TuRBOOptimizer:
    """TuRBO优化器控制器"""

    def __init__(self, num_subspaces=3, embedding_dim=384, length_init=0.8):
        self.num_subspaces = num_subspaces
        self.embedding_dim = embedding_dim
        self.length_init = length_init
        self.subspaces = []

        # 记录生成的候选解来自哪个子空间，用于后续归因更新
        # 格式: {prompt_str: subspace_index}
        self.candidate_map = {}

    def initialize_subspaces(self, population_embeddings):
        """
        初始化信任区域
        对应论文伪代码 Step 1 Line 5: 随机选择中心
        """
        self.subspaces = []
        indices = np.random.choice(len(population_embeddings), self.num_subspaces, replace=True)

        for idx in indices:
            center = population_embeddings[idx]
            tr = TuRBOSubspace(self.dim_check(center), center, length_init=self.length_init)
            self.subspaces.append(tr)

        logger.info(f"Initialized {len(self.subspaces)} Trust Regions")

    def dim_check(self, vec):
        return vec.shape[0]

    def restart_subspace(self, index, population_embeddings, population_objectives):
        """
        重启机制
        对应论文伪代码 Line 29: 在当前最优解附近重新初始化
        """
        # 找到当前种群中的最优解（这里简化为最大化其中一个目标，或者随机选择非支配解）
        # 为了鲁棒性，随机选择一个表现较好的个体作为新中心
        best_idx = np.random.randint(0, len(population_embeddings))
        new_center = population_embeddings[best_idx]

        self.subspaces[index] = TuRBOSubspace(
            self.dim_check(new_center),
            new_center,
            length_init=self.length_init
        )
        logger.info(f"Trust Region {index} Restarted.")

    def select_subspace_for_generation(self):
        """轮询或随机选择一个子空间用于生成"""
        return np.random.randint(0, self.num_subspaces)