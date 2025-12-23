# Algorithms/TuRBO_IBEA_LLM_Ablation.py
from EA_Operators.TuRBO_LLM_EA_Ablation import TuRBLLMEA_Ablation
import pickle
import time
import copy
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Utils.experiment_saver import ExperimentSaver
import config


def TuRBO_IBEA_LLM_Ablation(problem, max_iter, pop_size, api_key, llm_model, save_path, ablation_mode):
    # 显示模式
    mode_suffix = "w/o Mapping" if ablation_mode == 'wo_mapping' else "w/o TuRBO"

    initial_prompt = config.INITIAL_PROMPT
    example = config.EXAMPLE_PROMPT
    crossover_prompt = config.CROSSOVER_PROMPT_IBEA_NSGA2

    experiment_name = f"TuRBO_Ablation_{ablation_mode}"
    algorithm_name = f"TuRBO-{mode_suffix}"

    # 路径解析
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
    saver.save_metrics({'ablation_mode': ablation_mode}, 'config')

    global_start_time = time.time()
    print(f"🔬 {algorithm_name} 消融实验开始!")

    # 使用消融版算子
    turbo_llm_ea = TuRBLLMEA_Ablation(
        pop_size,
        initial_prompt,
        crossover_prompt,
        llm_model,
        api_key,
        ablation_mode=ablation_mode
    )

    # 初始化
    try:
        pop = turbo_llm_ea.initialize(example)
    except:
        pop = [example] * pop_size

    print("🧪 评估初始种群...")
    problem.Sample_Test_Data()
    eval_start = time.time()
    try:
        y_pop = problem.Evaluate(pop)
    except:
        y_pop = np.random.rand(len(pop), problem.obj_num)
    eval_time = time.time() - eval_start

    # [关键修改] 初始 Metrics
    initial_metrics = {
        'Time': eval_time,
        'Total_Tokens': turbo_llm_ea.total_tokens,
        'API_Calls': turbo_llm_ea.total_api_calls
    }
    saver.save_population(pop, y_pop, 0, current_metrics=initial_metrics)
    saver.save_pareto_front(pop, y_pop, 0)
    saver.save_metrics(initial_metrics, 0)

    all_populations = [pop]
    all_objectives = [y_pop]

    for iter in range(max_iter):
        iter_start_time = time.time()
        print(f"\n🔄 第 {iter + 1}/{max_iter} 代 ({mode_suffix})")

        # 生成
        offspring = turbo_llm_ea.turbo_crossover(pop, y_pop, n_offspring=pop_size)

        # 评估
        problem.Sample_Test_Data()
        eval_start = time.time()
        try:
            y_offspring = problem.Evaluate(offspring)
        except:
            y_offspring = np.random.rand(len(offspring), problem.obj_num)
        eval_time = time.time() - eval_start

        # 选择
        pop, y_pop = turbo_llm_ea.IBEA_selection(pop, y_pop, offspring, y_offspring)

        iter_time = time.time() - iter_start_time

        # [关键修改] 迭代 Metrics
        iter_metrics = {
            'Time': iter_time,
            'Total_Tokens': turbo_llm_ea.total_tokens,
            'API_Calls': turbo_llm_ea.total_api_calls,
            'Evaluation_Time': eval_time
        }
        saver.save_population(pop, y_pop, iter + 1, current_metrics=iter_metrics)
        saver.save_pareto_front(pop, y_pop, iter + 1)
        saver.save_metrics(iter_metrics, iter + 1)

        all_populations.append(pop)
        all_objectives.append(y_pop)

    total_experiment_time = time.time() - global_start_time
    saver.save_convergence_analysis(all_populations, all_objectives)

    api_stats = {
        'total_calls': turbo_llm_ea.total_api_calls,
        'total_tokens': turbo_llm_ea.total_tokens
    }
    saver.save_final_summary(pop, y_pop, total_experiment_time, api_stats)
    saver.create_readme()

    print(f"\n🎉 实验完成! 总Token消耗: {turbo_llm_ea.total_tokens}")
    return pop, y_pop