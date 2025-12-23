from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .utils import extract_edit_prompt, environment_selection, Tchebycheff, choice_matrix, IBEA_Selection
import json
import numpy as np
import random
import os
import time


def invoke_llm_with_tracking(llm, prompt_template, input_variables, tracker_obj):
    messages = prompt_template.format_messages(**input_variables)
    response = llm.invoke(messages)
    tracker_obj.total_api_calls += 1
    usage = response.response_metadata.get('token_usage', {})
    tokens = usage.get('total_tokens', 0)
    if tokens > 0:
        tracker_obj.total_tokens += tokens
    else:
        input_text = str(input_variables)
        output_text = response.content
        estimated = (len(input_text) + len(output_text)) // 4
        tracker_obj.total_tokens += estimated
    return response.content


class LLM_EA():
    def __init__(self, pop_size, initialize_prompt, crossover_prompt, llm_model, api_key):
        self.pop_size = pop_size
        self.total_api_calls = 0
        self.failed_api_calls = 0
        self.total_tokens = 0
        self.start_time = None

        # 初始化 LLM
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            self.llm_initialize = ChatZhipuAI(model="glm-4", api_key=api_key, temperature=0.7)
            self.llm_operator = ChatZhipuAI(model="glm-4", api_key=api_key, temperature=0.7)
        elif llm_model == 'deepseek-chat':
            self.llm_initialize = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1",
                                             model="deepseek-chat", temperature=0.7)
            self.llm_operator = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1",
                                           model="deepseek-chat", temperature=0.7)
        else:
            self.llm_initialize = ChatOpenAI(api_key=api_key, model=llm_model)
            self.llm_operator = ChatOpenAI(api_key=api_key, model=llm_model)

        # [关键修改] 增强 System Prompt，防止生成空内容
        system_instruction = (
            "You are an evolutionary operator for prompt optimization. "
            "Your task is to generate a new, improved prompt based on the input. "
            "CRITICAL: You MUST wrap your entire output prompt inside <START> and <END> tags. "
            "Do not output explanations, only the new prompt inside tags."
        )

        self.prompt_initialize = ChatPromptTemplate.from_messages([
            ("system", "You are an initializer. " + system_instruction),
            ("user", initialize_prompt)]
        )

        self.prompt_operator = ChatPromptTemplate.from_messages([
            ("system", "You are a crossover operator. " + system_instruction),
            ("user", crossover_prompt)]
        )

    # ... (initialize, crossover, naive, environment_selection, IBEA_selection 等方法保持之前的修复版逻辑)
    # 为节省篇幅，请确保这里保留了之前提供的 `invoke_llm_with_tracking` 调用逻辑
    # 核心是上面的 System Prompt 修改

    def initialize(self, example):
        pop = []
        self.start_time = time.time()
        print(f"🚀 开始初始化种群，需要生成 {self.pop_size} 个个体")
        for i in range(self.pop_size):
            start_call = time.time()
            while True:
                try:
                    output = invoke_llm_with_tracking(self.llm_initialize, self.prompt_initialize, {"example": example},
                                                      self)
                    call_time = time.time() - start_call
                    print(f"✅ 初始化个体 {i + 1}/{self.pop_size} 完成 | API调用时间: {call_time:.1f}秒")
                    break
                except Exception as e:
                    self.failed_api_calls += 1
                    time.sleep(5)
            individual = extract_edit_prompt(output)
            pop.extend(individual)
        return pop

    def crossover(self, pop):
        offsprings = []
        print(f"🧬 开始交叉变异，需要生成 {self.pop_size} 个后代")
        for i in range(self.pop_size):
            start_call = time.time()
            idx = np.random.choice(len(pop), 2, replace=False)
            while True:
                try:
                    output = invoke_llm_with_tracking(self.llm_operator, self.prompt_operator,
                                                      {"prompt1": pop[idx[0]], "prompt2": pop[idx[1]]}, self)
                    print(f"✅ 后代 {i + 1}/{self.pop_size} 生成完成")
                    break
                except Exception as e:
                    self.failed_api_calls += 1
                    time.sleep(5)
            offspring = extract_edit_prompt(output)
            offsprings.extend(offspring)
        return offsprings

    def enviromnent_selection(self, pop, y_pop, offspring, y_offspring):
        pop.extend(offspring)
        y_pop = np.concatenate((y_pop, y_offspring))
        pop_next, _, _, _ = environment_selection([pop, y_pop], self.pop_size)
        return pop_next[0], pop_next[1]

    def IBEA_selection(self, pop, y_pop, offspring, y_offspring):
        pop.extend(offspring)
        y_pop = np.concatenate((y_pop, y_offspring), axis=0)
        pop, y_pop = IBEA_Selection(pop, y_pop, self.pop_size, 0.05)
        return pop, y_pop


class LLM_MOEAD(LLM_EA):
    # MOEAD 类可以继承 LLM_EA，只需重写 init 和 evolution
    # 请确保引入之前提供的 MOEAD 修复逻辑（邻域修正等）
    def __init__(self, pop_size, obj_num, initialize_prompt, crossover_prompt, weight, num_sub_set, llm_model, api_key):
        super().__init__(pop_size, initialize_prompt, crossover_prompt, llm_model, api_key)
        self.weight = weight
        self.obj_num = obj_num

        # 邻域修正
        self.num_sub_set = min(num_sub_set, pop_size)

        from scipy.spatial.distance import cdist
        w_repeat1 = weight.reshape(1, self.pop_size, self.obj_num).repeat(self.pop_size, axis=0)
        w_repeat2 = weight.reshape(self.pop_size, 1, self.obj_num).repeat(self.pop_size, axis=1)
        dist = np.sqrt(np.sum((w_repeat1 - w_repeat2) ** 2, axis=2))
        self.B = np.argsort(dist, axis=1)[:, 0:self.num_sub_set]
        self.p_sel = np.ones([pop_size, self.num_sub_set]) / self.num_sub_set

    def evolution(self, pop, y_pop, obj_func):
        idx_choice = np.random.choice(self.pop_size, self.pop_size, replace=False)
        idx_sel = choice_matrix(self.p_sel, 2)
        # 修正索引范围
        max_idx = self.B.shape[1] - 1
        w_rand1 = self.B[idx_choice, np.clip(idx_sel[0, idx_choice], 0, max_idx)]
        w_rand2 = self.B[idx_choice, np.clip(idx_sel[1, idx_choice], 0, max_idx)]

        for i in range(self.pop_size):
            parent1 = pop[w_rand1[i]]
            parent2 = pop[w_rand2[i]]

            # 使用父类的 crossover 逻辑生成单个后代
            # 这里简化处理，直接调父类 crossover 可能会生成一批，MOEAD 只需要一个
            # 建议在 LLM_EA 中增加 crossover_single 方法，或者在这里手动调用
            try:
                output = invoke_llm_with_tracking(self.llm_operator, self.prompt_operator,
                                                  {"prompt1": parent1, "prompt2": parent2}, self)
                offspring = extract_edit_prompt(output)
            except:
                offspring = [parent1]

            if not offspring: offspring = [parent1]

            y_offspring = obj_func(offspring)
            if len(y_offspring.shape) == 1: y_offspring = y_offspring.reshape(1, -1)

            # Tchebycheff 更新逻辑 (同之前)
            z_min = np.min(np.vstack((y_pop, y_offspring)), axis=0)
            y_pop_tch = Tchebycheff(y_pop[self.B[idx_choice[i]]], self.weight[self.B[idx_choice[i]]])
            y_offspring_tch = Tchebycheff(y_offspring, self.weight[self.B[idx_choice[i]]])
            idx_update = self.B[idx_choice[i], y_offspring_tch < y_pop_tch]
            for idx in idx_update:
                pop[idx] = offspring[0]
                y_pop[idx] = y_offspring

        return pop, y_pop