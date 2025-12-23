# EA_Operators/TuRBO_LLM_EA.py
from .LLM_EA import LLM_EA, invoke_llm_with_tracking
from .embedding_utils import RobustPromptEmbedder
from .TuRBO import TuRBOOptimizer
import numpy as np
import logging
import os

# LangChain 相关导入
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 如果没有安装 langchain_community 或不需要 ChatZhipuAI，可以注释掉下面这行
try:
    from langchain_community.chat_models import ChatZhipuAI
except ImportError:
    ChatZhipuAI = None

logger = logging.getLogger(__name__)


class TuRBLLMEA(LLM_EA):
    """
    集成TuRBO的LLM进化算法
    实现论文 3.3.1 Hybrid Candidate Generation Strategy
    """

    def __init__(self, pop_size, initialize_prompt, crossover_prompt, llm_model, api_key,
                 turbo_subspaces=3, use_embedding=True, embedding_model='all-MiniLM-L6-v2'):
        # 1. 调用父类初始化
        super().__init__(pop_size, initialize_prompt, crossover_prompt, llm_model, api_key)

        # =======================================================
        # 初始化 TuRBO 专用 LLM
        # =======================================================
        if llm_model == 'glm':
            if ChatZhipuAI is None:
                raise ImportError("ChatZhipuAI 未导入，请检查 langchain_community 是否安装")
            os.environ["ZHIPUAI_API_KEY"] = api_key
            self.llm = ChatZhipuAI(model="glm-4", temperature=0.7)
        elif llm_model == 'deepseek-chat':
            self.llm = ChatOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
                temperature=0.7
            )
        else:
            # 默认为 GPT 或其他 OpenAI 兼容模型
            self.llm = ChatOpenAI(api_key=api_key, model=llm_model, temperature=0.7)

        print(f"✅ TuRBO LLM Client Initialized: {self.llm}")

        self.use_embedding = use_embedding
        self.embedder = RobustPromptEmbedder(embedding_model)
        self.prompt_embeddings = {}  # 缓存: str -> np.array

        # TuRBO 核心组件
        self.turbo_opt = TuRBOOptimizer(
            num_subspaces=turbo_subspaces,
            embedding_dim=self.embedder.embedding_dim
        )
        self.turbo_initialized = False

        # 初始化 TuRBO 变异 Prompt 模板
        # 增强了 System Prompt 以确保输出格式正确
        self.turbo_mutation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in optimizing recommendation prompts. "
                       "CRITICAL: You MUST wrap your entire output prompt inside <START> and <END> tags. "
                       "Do not output explanations, only the new prompt inside tags."),
            ("user",
             "I have a prompt for a recommendation task. I want to modify this prompt to explore a specific semantic direction.\n"
             "The prompt is: \n{anchor_prompt}\n\n"
             "Please rewrite this prompt to maintain its core logic but change the wording or structure to potentially improve recommendation accuracy and diversity.\n"
             "The new prompt must be wrapped with <START> and <END>.")
        ])

    def get_embedding(self, prompt):
        if prompt not in self.prompt_embeddings:
            self.prompt_embeddings[prompt] = self.embedder.encode(prompt)[0]
        return self.prompt_embeddings[prompt]

    def update_embeddings_batch(self, population):
        """批量更新缓存"""
        embeddings = []
        for p in population:
            embeddings.append(self.get_embedding(p))
        return np.array(embeddings)

    def turbo_generation(self, population, n_offspring):
        """
        TuRBO 混合生成流程
        对应论文 Algorithm 2, Step 1 (Lines 9-14)
        """
        # 1. 首次运行时初始化子空间
        pop_embeddings = self.update_embeddings_batch(population)
        if not self.turbo_initialized:
            self.turbo_opt.initialize_subspaces(pop_embeddings)
            self.turbo_initialized = True

        offspring = []
        offspring_source_map = []  # 记录 (offspring_index, subspace_index) 以便后续更新

        print("🔄 TuRBO Generation Steps: Sampling -> Anchor -> Mutation")

        for i in range(n_offspring):
            # Step 1: 选择子空间并采样
            subspace_idx = self.turbo_opt.select_subspace_for_generation()
            subspace = self.turbo_opt.subspaces[subspace_idx]

            # z ~ N(ck, Lk)
            z_vector = subspace.sample_vector()

            # Step 2: 语义锚点选择
            anchor_prompt = self._find_semantic_anchor(population, pop_embeddings, z_vector)

            # Step 3: 离散文本实现 (LLM Mutation)
            try:
                new_prompt = self._generate_discrete_text(anchor_prompt)
                offspring.append(new_prompt)
                offspring_source_map.append(subspace_idx)  # 记录这个后代是由哪个 TR 生成的
            except Exception as e:
                logger.error(f"LLM Generation failed: {e}")
                # 失败回退
                offspring.append(anchor_prompt)
                offspring_source_map.append(subspace_idx)

        return offspring, offspring_source_map

    def _find_semantic_anchor(self, population, embeddings, z_vector):
        """
        计算 Cosine Similarity 并选择 Anchor
        """
        norm_z = np.linalg.norm(z_vector)
        if norm_z == 0: return np.random.choice(population)

        # 批量计算相似度
        dot_products = np.dot(embeddings, z_vector)
        norms = np.linalg.norm(embeddings, axis=1)
        similarities = dot_products / (norms * norm_z + 1e-9)

        best_idx = np.argmax(similarities)
        return population[best_idx]

    def _generate_discrete_text(self, anchor_prompt):
        """
        调用 LLM 进行定向变异
        [关键修复] 使用 invoke_llm_with_tracking 来捕获 Token
        """
        response = invoke_llm_with_tracking(
            self.llm,
            self.turbo_mutation_prompt,
            {"anchor_prompt": anchor_prompt},
            self  # 传入 self 以更新 total_tokens
        )
        extracted = self.extract_edit_prompt(response)
        if extracted:
            return extracted[0]
        return anchor_prompt

    def extract_edit_prompt(self, response):
        """提取 Prompt 内容"""
        import re
        patterns = [
            r'<START>\s*(.*?)\s*<END>',
            r'```(?:python)?\s*(.*?)\s*```',
            r'["\']([^"\']*?)["\']',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                cleaned = [match.strip() for match in matches if match.strip()]
                if cleaned:
                    return cleaned
        return [response.strip()] if response.strip() else []

    def update_turbo_regions(self, offspring_list, offspring_objs, offspring_source_map, archive_hv, calculator):
        """
        更新信任区域状态
        对应论文 Algorithm 2, Step 3 (Lines 20-30)
        """
        # 这里需要实现具体的更新逻辑，或者保持为空如果暂时不需要自适应调整
        pass

    def fallback_crossover_batch(self, pop, n_offspring):
        """
        如果 TuRBO 失败，回退到简单的随机选择
        """
        import random
        return [random.choice(pop) for _ in range(n_offspring)]