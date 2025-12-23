from EA_Operators.LLM_EA import LLM_EA
import pickle
import time
import copy
from datetime import datetime, timedelta
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Utils.experiment_saver import ExperimentSaver
import config


def NSGA2_LLM(problem, max_iter, pop_size, api_key, llm_model, save_path):
    # ... (参数设置保持不变)
    initial_prompt = config.INITIAL_PROMPT
    example = config.EXAMPLE_PROMPT
    crossover_prompt = config.CROSSOVER_PROMPT_IBEA_NSGA2

    experiment_name = f"NSGA2_LLM_Experiment"
    algorithm_name = "NSGA2-LLM"

    # 路径解析逻辑 (保持不变)
    path_parts = save_path.split('/')
    if len(path_parts) >= 3:
        dataset_name = path_parts[-2]
        file_name = path_parts[-1]
        if 'Acc_Div_Fair' in file_name:
            objectives_name = 'Acc_Div_Fair'
        elif 'Acc_Div' in file_name:
            objectives_name = 'Acc_Div'
        elif 'Acc_Fair' in file_name:
            objectives_name = 'Acc_Fair'
        else:
            objectives_name = 'Unknown'
    else:
        dataset_name, objectives_name = "Unknown", "Unknown"

    saver = ExperimentSaver(experiment_name, algorithm_name, dataset_name, objectives_name, config.SEED)

    # 保存Config
    saver.save_metrics({'population_size': pop_size, 'llm_model': llm_model}, 'config')

    # Initialization
    global_start_time = time.time()
    print(f"🔬 NSGA2-LLM 实验开始!")

    llm_ea = LLM_EA(pop_size, initial_prompt, crossover_prompt, llm_model, api_key)
    pop = llm_ea.initialize(example)

    print("🧪 开始评估初始种群...")
    problem.Sample_Test_Data()
    eval_start = time.time()
    y_pop = problem.Evaluate(pop)
    eval_time = time.time() - eval_start

    # [关键修改] 构建初始 metrics，包含 Total_Tokens
    initial_metrics = {
        'Time': eval_time,
        'Total_Tokens': llm_ea.total_tokens,  # <--- 获取 Token
        'API_Calls': llm_ea.total_api_calls
    }

    # [关键修改] 传递 metrics 给 saver
    saver.save_population(pop, y_pop, 0, current_metrics=initial_metrics)
    saver.save_pareto_front(pop, y_pop, 0)
    saver.save_metrics(initial_metrics, 0)

    all_populations = [pop]
    all_objectives = [y_pop]

    for iter in range(max_iter):
        iter_start_time = time.time()
        print(f"\n🔄 第 {iter + 1}/{max_iter} 代开始")

        offspring = llm_ea.crossover(pop)

        problem.Sample_Test_Data()
        eval_start = time.time()
        y_offspring = problem.Evaluate(offspring)
        eval_time = time.time() - eval_start

        pop, y_pop = llm_ea.enviromnent_selection(pop, y_pop, offspring, y_offspring)

        # 迭代统计
        iter_time = time.time() - iter_start_time

        # [关键修改] 构建迭代 metrics
        iter_metrics = {
            'Time': iter_time,
            'Total_Tokens': llm_ea.total_tokens,  # <--- 实时累积值
            'API_Calls': llm_ea.total_api_calls,
            'Evaluation_Time': eval_time
        }

        # [关键修改] 传递 metrics
        saver.save_population(pop, y_pop, iter + 1, current_metrics=iter_metrics)
        pareto_data = saver.save_pareto_front(pop, y_pop, iter + 1)
        saver.save_metrics(iter_metrics, iter + 1)

        all_populations.append(pop)
        all_objectives.append(y_pop)

        print(f"📈 统计: 耗时 {iter_time:.1f}s | Tokens {llm_ea.total_tokens}")

    total_experiment_time = time.time() - global_start_time
    saver.save_convergence_analysis(all_populations, all_objectives)

    api_stats = {
        'total_calls': llm_ea.total_api_calls,
        'total_tokens': llm_ea.total_tokens,
        'failed_calls': llm_ea.failed_api_calls
    }
    saver.save_final_summary(pop, y_pop, total_experiment_time, api_stats)
    saver.create_readme()

    print(f"\n🎉 实验完成! 总Token消耗: {llm_ea.total_tokens}")
    return pop, y_pop