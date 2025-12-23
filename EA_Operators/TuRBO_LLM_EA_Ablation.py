# EA_Operators/TuRBO_LLM_EA_Ablation.py
from .TuRBO_LLM_EA import TuRBLLMEA
from .embedding_utils import DummyEmbedder, RobustPromptEmbedder
import logging

logger = logging.getLogger(__name__)


class TuRBLLMEA_Ablation(TuRBLLMEA):
    """
    用于消融实验的类，继承自 TuRBLLMEA。
    不修改原文件，通过 ablation_mode 参数控制逻辑。
    """

    def __init__(self, pop_size, initialize_prompt, crossover_prompt, llm_model, api_key,
                 turbo_subspaces=3, use_embedding=True, embedding_model='all-MiniLM-L6-v2',
                 ablation_mode='none'):
        """
        :param ablation_mode:
            - 'wo_mapping': 禁用语义映射 (使用随机/Dummy嵌入)，但保留TuRBO逻辑
            - 'wo_turbo': 保留语义映射，但禁用TuRBO信任区域逻辑 (回退到普通交叉)
        """
        # 先调用父类初始化
        # 注意：父类中 turbo_enabled 默认绑定在 use_embedding 上，我们需要在下面手动覆盖
        super().__init__(pop_size, initialize_prompt, crossover_prompt, llm_model, api_key,
                         turbo_subspaces, use_embedding, embedding_model)

        self.ablation_mode = ablation_mode

        # === 实现消融逻辑 ===
        if ablation_mode == 'wo_mapping':
            logger.info("🔧 [Ablation] Mode: w/o Mapping (禁用语义嵌入)")
            # 1. 强制使用 DummyEmbedder (随机向量)
            self.embedder = DummyEmbedder()
            self.use_embedding = False
            # 2. 强制开启 TuRBO (即使没有真实嵌入，也要让TuRBO在随机空间跑，以证明语义的重要性)
            self.turbo_enabled = True
            # 清空旧缓存
            self.prompt_embeddings = {}

        elif ablation_mode == 'wo_turbo':
            logger.info("🔧 [Ablation] Mode: w/o TuRBO (禁用信任区域)")
            # 1. 保持 use_embedding = True (父类已处理)，保留语义信息用于分析
            # 2. 强制关闭 TuRBO
            self.turbo_enabled = False

        else:
            logger.info("🔧 [Ablation] Mode: Standard (无消融)")

    def turbo_crossover(self, pop, y_pop, n_offspring=None):
        """
        重写交叉方法以支持日志输出
        """
        if self.ablation_mode == 'wo_turbo':
            # 如果是 w/o TuRBO 模式，直接调用回退方法（普通交叉）
            print("⚠️ [Ablation] TuRBO已禁用，执行常规交叉...")
            return self.fallback_crossover_batch(pop, n_offspring if n_offspring else self.pop_size)

        # 否则 (包括 w/o mapping 和 standard)，调用父类的 TuRBO 逻辑
        # w/o mapping 时，父类逻辑会基于 DummyEmbedder 产生的随机向量运行
        return super().turbo_crossover(pop, y_pop, n_offspring)