import numpy as np
import random
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatZhipuAI # 如需使用智谱请取消注释
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
import os
import pickle
import time
from Utils import nondomination
import Utils.hypervolume as hypervolume
import copy
import ast
from fuzzywuzzy import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed

# =================配置区域=================
# 根据您的API限流情况调整。M3 Pro处理本地逻辑很快，瓶颈在网络IO。
# 如果是 DeepSeek/OpenAI，通常可以开到 10-20。
MAX_WORKERS = 10


# =========================================

# Basic Functions
#####################################################################
def extract_item_list(response, target):
    try:
        response = response.replace(" ", " ")
        target = target.replace(" ", " ").replace("&amp;", "&").replace("&reg;", "®")
        index = response.rfind(target)
        if index != -1:
            preceding_text = response[:index].strip()
            numbers = re.findall(r'\d+', preceding_text)
            if numbers:
                result_list = numbers
            else:
                result_list = []
        else:
            result_list = []
    except:
        result_list = []
    return result_list


def detect_error(response, target, mode='improve'):
    """
    安全地检测目标项在响应列表中的位置
    """
    try:
        # 检查输入有效性
        if not response or not isinstance(response, list) or len(response) == 0:
            return False, None

        if not target:
            return False, None

        target_str = str(target).strip().lower()

        # 规范化响应列表
        normalized_response = []
        for item in response:
            try:
                item_str = str(item).strip().lower()
                if item_str:
                    normalized_response.append(item_str)
            except:
                continue

        if len(normalized_response) == 0:
            return False, None

        # 查找目标
        for idx, item in enumerate(normalized_response):
            # 精确匹配 或 包含匹配 或 模糊匹配
            if (item == target_str or
                    target_str in item or
                    item in target_str or
                    fuzz.ratio(item, target_str) > 80):
                return True, idx

        return False, None

    except Exception as e:
        # print(f"⚠️ detect_error 函数出错: {e}")
        return False, None


def diversity_calculate(list_recommond, sample_data):
    if not list_recommond or len(list_recommond) == 0:
        return 0.0

    record_category = []
    for product in list_recommond:
        try:
            if "candidate_set" in sample_data and "category_list" in sample_data:
                # 尝试找到产品在候选集中的索引
                if product in sample_data["candidate_set"]:
                    index = sample_data["candidate_set"].index(product)
                    category = sample_data["category_list"][index]
                    record_category.extend(category)
        except (ValueError, IndexError, TypeError):
            continue

    if len(record_category) == 0:
        return 0.0

    unique_category = list(set(record_category))
    diversity = len(unique_category) / len(record_category)
    return min(diversity, 1.0)


def APT(list_recommond, original_data):
    if not list_recommond or len(list_recommond) == 0:
        return 0.0

    record_set_label = []
    for product in list_recommond:
        try:
            if "candidate_set" in original_data and "popular_list" in original_data:
                if product in original_data["candidate_set"]:
                    idx = original_data["candidate_set"].index(product)
                    record_set_label.append(original_data["popular_list"][idx])
        except (ValueError, IndexError, TypeError):
            continue

    if len(record_set_label) == 0:
        return 0.0

    vec_set_label = np.array(record_set_label)
    # 计算流行度偏见 (Average Popularity of Top-k)
    # 这里假设 label=1 是流行物品，我们希望推荐更多非流行物品(0)，还是根据具体定义？
    # 原代码逻辑：np.sum(vec_set_label) / len(list_recommond)
    # 这计算的是流行物品的比例。如果是Fairness目标，通常希望这个值适中或较低。
    fairness = np.sum(vec_set_label) / len(list_recommond)
    return min(fairness, 1.0)


# Base Evaluator Class with Parallel Processing
#####################################################################
class BaseEvaluator:
    def __init__(self, train_data, batch_num, api_key, llm_model='deepseek-chat'):
        self.train_data = train_data
        self.batch_num = batch_num
        self.api_key = api_key
        self.llm_model = llm_model
        self.sample_data = []

        # 初始化 LLM
        if llm_model == 'glm':
            # os.environ["ZHIPUAI_API_KEY"] = api_key
            from langchain_community.chat_models import ChatZhipuAI
            self.llm_recommond = ChatZhipuAI(model="glm-4", api_key=api_key, temperature=0.7)
            self.llm_translate = ChatZhipuAI(model="glm-4", api_key=api_key, temperature=0.1)
        else:
            base_url = "https://api.deepseek.com/v1" if 'deepseek' in llm_model else None
            self.llm_recommond = ChatOpenAI(api_key=api_key, model=llm_model, base_url=base_url, temperature=0.7)
            self.llm_translate = ChatOpenAI(api_key=api_key, model=llm_model, base_url=base_url, temperature=0.1)

        self.output_parser = StrOutputParser()

        # 初始化翻译链 (用于解析列表)
        prompt_translate = ChatPromptTemplate.from_messages([
            ("user", '''Please transfer a set of product names into a Python list. 
            Input: {input}
            Output (Python List only):''')
        ])
        self.chain_translate = prompt_translate | self.llm_translate | self.output_parser

    def Sample_Test_Data(self):
        """随机采样测试数据"""
        self.sample_data = random.sample(self.train_data, self.batch_num)

    def Translate(self, input_text):
        """解析LLM输出为Python列表"""
        # 1. 尝试直接使用 ast 解析
        cleaned_response = input_text.strip()

        # 提取 markdown 代码块
        matches = re.findall(r"```(?:python)?\s*(.*?)\s*```", cleaned_response, re.DOTALL)
        if matches:
            cleaned_response = matches[0].strip()

        # 方法1: ast
        try:
            if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
                result = ast.literal_eval(cleaned_response)
                if isinstance(result, list): return result
        except:
            pass

        # 方法2: 正则提取引号内容
        try:
            items = re.findall(r'"([^"]*)"', cleaned_response)
            if items: return items
        except:
            pass

        try:
            items = re.findall(r"'([^']*)'", cleaned_response)
            if items: return items
        except:
            pass

        # 方法3: LLM 辅助解析 (如果上述都失败)
        try:
            response = self.chain_translate.invoke({"input": input_text})
            # 递归调用一次 ast 解析翻译后的结果
            try:
                res_clean = response.strip()
                if res_clean.startswith('[') and res_clean.endswith(']'):
                    return ast.literal_eval(res_clean)
            except:
                pass
        except Exception as e:
            # print(f"Translate Error: {e}")
            pass

        return []

    def _process_single_sample(self, data, chain_recommond):
        """处理单个样本的内部函数，用于并行调用"""
        res = {'reward': 0.0, 'diversity': 0.0, 'fairness': 0.0}

        try:
            # 调用LLM推荐
            response = chain_recommond.invoke({"samples": data["input"]})

            # 解析结果
            parsed_response = self.Translate(response)

            if not parsed_response or not isinstance(parsed_response, list) or len(parsed_response) == 0:
                return res

            parsed_response = [str(item).strip() for item in parsed_response]

            # 1. 计算 Accuracy (Reward)
            flag_error, target_index = detect_error(parsed_response, data['target'], mode='select')
            if flag_error and target_index is not None and target_index >= 0:
                # MRR (Mean Reciprocal Rank)
                res['reward'] = 1.0 / (target_index + 1)

            # 2. 计算 Diversity
            res['diversity'] = diversity_calculate(parsed_response, data)

            # 3. 计算 Fairness
            res['fairness'] = APT(parsed_response, data)

            return res

        except Exception as e:
            # print(f"Sample Eval Error: {e}")
            return res

    def Evaluate_(self, prompt):
        """并行评估函数"""
        prompt_recommond = ChatPromptTemplate.from_messages([
            ("system", "You are a recommender system."),
            ("user", prompt + "\n\nUser History:\n{samples}\n\nRecommended Items:")
        ])
        chain_recommond = prompt_recommond | self.llm_recommond | self.output_parser

        rewards = []
        diversities = []
        fairnesses = []

        # print(f"   [Parallel] Processing {len(self.sample_data)} samples...")

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(self._process_single_sample, data, chain_recommond) for data in self.sample_data]

            for future in as_completed(futures):
                try:
                    res = future.result()
                    rewards.append(res['reward'])
                    diversities.append(res['diversity'])
                    fairnesses.append(res['fairness'])
                except Exception as e:
                    # print(f"Future Error: {e}")
                    pass

        # 计算平均值
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        avg_diversity = sum(diversities) / len(diversities) if diversities else 0.0
        avg_fairness = sum(fairnesses) / len(fairnesses) if fairnesses else 0.0

        return avg_reward, avg_diversity, avg_fairness


# Specific Problem Classes
#####################################################################

class Acc_Div_Fair(BaseEvaluator):
    def __init__(self, train_data, batch_num, api_key, llm_model='deepseek-chat'):
        super().__init__(train_data, batch_num, api_key, llm_model)
        self.obj_num = 3

    def Evaluate(self, pop):
        f = []
        for i, prompt in enumerate(pop):
            # print(f"Evaluating Prompt {i+1}/{len(pop)}...")
            r, d, fa = self.Evaluate_(prompt)
            f.append([r, d, fa])
        return np.array(f)


class Acc_Div(BaseEvaluator):
    def __init__(self, train_data, batch_num, api_key, llm_model='deepseek-chat'):
        super().__init__(train_data, batch_num, api_key, llm_model)
        self.obj_num = 2

    def Evaluate(self, pop):
        f = []
        for i, prompt in enumerate(pop):
            r, d, fa = self.Evaluate_(prompt)
            f.append([r, d])  # 只返回 Acc 和 Div
        return np.array(f)


class Acc_Fair(BaseEvaluator):
    def __init__(self, train_data, batch_num, api_key, llm_model='deepseek-chat'):
        super().__init__(train_data, batch_num, api_key, llm_model)
        self.obj_num = 2

    def Evaluate(self, pop):
        f = []
        for i, prompt in enumerate(pop):
            r, d, fa = self.Evaluate_(prompt)
            f.append([r, fa])  # 只返回 Acc 和 Fair
        return np.array(f)