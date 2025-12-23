from EA_Operators.LLM_EA import LLM_MOEAD
from pymoo.util.ref_dirs import get_reference_directions
import pickle
import time
import copy
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Utils.experiment_saver import ExperimentSaver
import config


def MOEAD_LLM(problem, max_iter, pop_size, num_sub_set, api_key, llm_model, save_path):
    initial_prompt = config.INITIAL_PROMPT
    example = config.EXAMPLE_PROMPT
    crossover_prompt = config.CROSSOVER_PROMPT_MOEAD

    experiment_name = f"MOEAD_LLM_Experiment"
    algorithm_name = "MOEAD-LLM"

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
    saver.save_metrics({'population_size': pop_size}, 'config')

    global_start_time = time.time()
    print(f"🔬 MOEA/D-LLM 实验开始!")

    # 权重生成逻辑 (保持不变)
    obj_num = getattr(problem, 'obj_num', 2)
    try:
        weights = get_reference_directions("energy", obj_num, pop_size)
    except:
        if obj_num == 2:
            weights = np.array([[i / (pop_size - 1), 1 - i / (pop_size - 1)] for i in range(pop_size)])
        else:
            weights = np.ones((pop_size, obj_num)) / obj_num

    llm_ea = LLM_MOEAD(pop_size, obj_num, initial_prompt, crossover_prompt, weights, num_sub_set, llm_model, api_key)
    pop = llm_ea.initialize(example)

    print("🧪 开始评估初始种群...")
    problem.Sample_Test_Data()
    eval_start = time.time()
    y_pop = problem.Evaluate(pop)
    eval_time = time.time() - eval_start

    # [关键修改]
    initial_metrics = {
        'Time': eval_time,
        'Total_Tokens': llm_ea.total_tokens,
        'API_Calls': llm_ea.total_api_calls
    }
    saver.save_population(pop, y_pop, 0, current_metrics=initial_metrics)
    saver.save_pareto_front(pop, y_pop, 0)
    saver.save_metrics(initial_metrics, 0)

    all_populations = [pop]
    all_objectives = [y_pop]

    for iter in range(max_iter):
        iter_start_time = time.time()
        print(f"\n🔄 第 {iter + 1}/{max_iter} 代开始")

        problem.Sample_Test_Data()
        pop, y_pop = llm_ea.evolution(pop, y_pop, problem.Evaluate)

        iter_time = time.time() - iter_start_time

        # [关键修改]
        iter_metrics = {
            'Time': iter_time,
            'Total_Tokens': llm_ea.total_tokens,
            'API_Calls': llm_ea.total_api_calls
        }
        saver.save_population(pop, y_pop, iter + 1, current_metrics=iter_metrics)
        saver.save_pareto_front(pop, y_pop, iter + 1)
        saver.save_metrics(iter_metrics, iter + 1)

        all_populations.append(pop)
        all_objectives.append(y_pop)

    total_experiment_time = time.time() - global_start_time
    saver.save_convergence_analysis(all_populations, all_objectives)

    api_stats = {'total_calls': llm_ea.total_api_calls, 'total_tokens': llm_ea.total_tokens}
    saver.save_final_summary(pop, y_pop, total_experiment_time, api_stats)
    saver.create_readme()

    print(f"\n🎉 实验完成! 总Token消耗: {llm_ea.total_tokens}")
    return pop, y_pop