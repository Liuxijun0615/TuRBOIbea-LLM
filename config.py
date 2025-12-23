# config.py
"""
RSBench 实验配置文件
集中管理所有实验参数，避免重复和导入冲突
"""

# ==================== LLM 配置 ====================
# API 密钥配置
OPENAI_KEY = 'sk-ffb8c5feaed94f1a9021941721e2aba0'  # 替换为你的实际API密钥“lxj”

# LLM 模型配置
LLM_MODEL = 'deepseek-chat'
LLM_BASE_URL = "https://api.deepseek.com/v1"

# LLM 参数
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000

# ==================== 实验配置 ====================
# 随机种子
SEED = 625

# 进化算法参数
MAX_ITERATIONS = 3   # 10
POPULATION_SIZE = 3    # 10
BATCH_SIZE = 3     # 10

# MOEA/D 特定参数
NEIGHBORHOOD_SIZE = 1  # 邻域大小，通常设置为种群大小的20-30%

# 调试模式
DEBUG_MODE = True
LOG_LEVEL = 'DEBUG'  # DEBUG, INFO, WARNING, ERROR

# 评估参数
MAX_RETRIES = 3
RETRY_DELAY = 5

# ==================== 数据集和目标配置 ====================
# 所有要运行的实验组合
# DATA_OBJECTIVES = [
#     ['Movie', 'Acc_Div'],
#     ['Game', 'Acc_Div'],
#     ['Bundle', 'Acc_Div'],
#     ['Movie', 'Acc_Fair'],
#     ['Game', 'Acc_Fair'],
#     ['Bundle', 'Acc_Fair'],
#     ['Movie', 'Acc_Div_Fair'],
#     ['Game', 'Acc_Div_Fair'],
#     ['Bundle', 'Acc_Div_Fair']
# ]
DATA_OBJECTIVES = [
    ['Bundle', 'Acc_Div_Fair'],
    ['Bundle', 'Acc_Div'],
    ['Bundle', 'Acc_Fair']
]
# 数据集路径配置
DATASET_PATHS = {
    'Movie': 'Dataset/Movie',
    'Game': 'Dataset/Game',
    'Bundle': 'Dataset/Bundle'
}

# ==================== 结果保存配置 ====================
RESULTS_BASE_DIR = 'Results'

# ==================== 算法特定配置 ====================
# IBEA 配置
IBEA_KAPPA = 0.05

# MOEA/D 配置
MOEAD_CONFIG = {
    'neighborhood_size': 3,
    'weight_generation': 'energy',  # 权重向量生成方法
    'decomposition_method': 'tchebycheff',  # 分解方法
    'neighborhood_selection_prob': 0.9  # 从邻域中选择父代的概率
}

# TuRBO配置
TURBO_CONFIG = {
    'use_embedding': False,
    'embedding_model': 'all-MiniLM-L6-v2',
    'turbo_subspaces': 3,
    'fallback_enabled': True,
    'robust_mode': True
}

# 提示模板配置
INITIAL_PROMPT = (
    "Now, I have a prompt for my task. I want to modify this prompt to better achieve my task. \n"
    "I will give an example of my current prompt. Please randomly generate a prompt based on my example. \n"
    "My example is as follows: \n"
    "{example} \n"
    "Note that the final prompt should be bracketed with <START> and <END>."
)

EXAMPLE_PROMPT = (
    "Based on the user's current session interactions, you need to answer the following subtasks step by step:\n"
    "1. Discover combinations of items within the session, where the number of combinations can be one or more.\n"
    "2. Based on the items within each combination, infer the user's interactive intent within each combination.\n"
    "3. Select the intent from the inferred ones that best represents the user's current preferences.\n"
    "4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.\n"
    "Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set.\n"
)

CROSSOVER_PROMPT_IBEA_NSGA2 = (
    "Please follow the instruction step-by-step to generate a better prompt. \n"
    "1. Cross over the following prompts and generate two new prompts: \n"
    "Prompt 1: {prompt1} \n"
    "Prompt 2: {prompt2}. \n"
    "2. Mutate the prompt generated in Step 1 and generate "
    "a final prompt bracketed with <START> and <END>."
)

CROSSOVER_PROMPT_MOEAD = (
    "Please follow the instruction step-by-step to generate a better prompt. \n"
    "1. Cross over the following prompts and generate a new prompt: \n"
    "Prompt 1: {prompt1} \n"
    "Prompt 2: {prompt2}. \n"
    "2. Mutate the prompt generated in Step 1 and generate "
    "a final prompt bracketed with <START> and <END>."
)


# ==================== 工具函数 ====================
def get_save_path(algorithm, dataset, objectives, seed=SEED):
    """生成结果保存路径"""
    import os
    from datetime import datetime

    # 基础路径
    base_dir = f"{RESULTS_BASE_DIR}/{dataset}"
    os.makedirs(base_dir, exist_ok=True)

    # 带时间戳的详细路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_dir = f"{base_dir}/{algorithm}_{objectives}_Seed_{seed}_{timestamp}"
    os.makedirs(detailed_dir, exist_ok=True)

    # 兼容性路径（原有格式）
    compatibility_path = f"{base_dir}/{algorithm}_{objectives}_Seed_{seed}"

    return detailed_dir, compatibility_path


def get_dataset_path(dataset_name, seed=SEED, file_type="train"):
    """获取数据集文件路径"""
    base_path = DATASET_PATHS.get(dataset_name, f"Dataset/{dataset_name}")

    if file_type == "train":
        return f"{base_path}/train_seed_{seed}.json"
    elif file_type == "validation":
        return f"{base_path}/valid.json"
    else:
        return f"{base_path}/{file_type}"