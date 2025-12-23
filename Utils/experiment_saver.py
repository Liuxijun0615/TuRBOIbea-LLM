import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from Utils import nondomination
from Utils.hypervolume import hypervolume


class ExperimentSaver:
    def __init__(self, experiment_name, algorithm_name, dataset_name, objectives, seed):
        self.experiment_name = experiment_name
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name
        self.objectives = objectives
        self.seed = seed

        # 1. 自动解析目标名称
        self.obj_names = self._map_objective_names(objectives)
        self.n_obj = len(self.obj_names)

        # 2. 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"Results/{dataset_name}/{algorithm_name}_{objectives}_Seed_{seed}_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        # 3. 初始化追踪记录
        self.progress_file = os.path.join(self.save_dir, "progress_summary.csv")
        self.history_records = []

        # 4. 保存实验配置
        self.config = {
            'experiment_name': experiment_name,
            'algorithm': algorithm_name,
            'dataset': dataset_name,
            'objectives_str': objectives,
            'objective_names': self.obj_names,
            'seed': seed,
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'save_dir': self.save_dir
        }
        self._save_config()

    def _map_objective_names(self, objectives_str):
        mapping = []
        if 'Acc' in objectives_str: mapping.append('Accuracy')
        if 'Div' in objectives_str: mapping.append('Diversity')
        if 'Fair' in objectives_str: mapping.append('Fairness')
        if not mapping:
            return [f'Obj_{i + 1}' for i in range(objectives_str.count('_') + 1)]
        return mapping

    def _save_config(self):
        config_file = f"{self.save_dir}/config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    # =======================================================
    # [修改] 允许 save_population 接收 current_metrics
    # =======================================================
    def save_population(self, population, objectives, iteration, current_metrics=None):
        """
        兼容接口：直接转发给 save_iteration_data
        """
        return self.save_iteration_data(population, objectives, iteration, current_metrics=current_metrics)

    def save_iteration_data(self, population, objectives, iteration, current_metrics=None, additional_info=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. 计算 HV
        try:
            hv_value = hypervolume(objectives, ref_point=np.zeros(self.n_obj))
        except Exception:
            hv_value = 0.0

        # 2. 保存详细数据 (CSV + Pickle)
        df_pop = pd.DataFrame(objectives, columns=self.obj_names)
        df_pop['Prompt'] = population
        df_pop['Iteration'] = iteration

        # 保存 CSV
        csv_file = f"{self.save_dir}/population_iter_{iteration:02d}.csv"
        df_pop.to_csv(csv_file, index=False, encoding='utf-8')

        # 保存 Pickle
        pop_data = {
            'iteration': iteration,
            'population': population,
            'objectives': objectives,
            'hypervolume': hv_value,
            'metrics': current_metrics,
            'timestamp': timestamp
        }
        if additional_info: pop_data.update(additional_info)
        pickle_file = f"{self.save_dir}/population_iter_{iteration:02d}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(pop_data, f)

        # 3. [关键] 更新 progress_summary.csv
        stats = {
            'Iteration': iteration,
            'Hypervolume': hv_value,
            'Timestamp': timestamp
        }

        # 将传入的 metrics (包含 Total_Tokens) 合并到 stats 中
        if current_metrics:
            stats.update(current_metrics)

        # 添加目标均值
        for col in self.obj_names:
            stats[f'{col}_Mean'] = df_pop[col].mean()
            stats[f'{col}_Max'] = df_pop[col].max()

        self.history_records.append(stats)

        # 实时写入 CSV
        pd.DataFrame(self.history_records).to_csv(self.progress_file, index=False)

        print(f"💾 Iter {iteration}: HV={hv_value:.4f} | Tokens={stats.get('Total_Tokens', 0)} | Data saved.")
        return hv_value

    def save_pareto_front(self, population, objectives, iteration):
        if len(objectives) == 0: return None
        try:
            fronts = nondomination.fast_non_dominated_sort(objectives)
            if not fronts or len(fronts[0]) == 0: return None
            pareto_indices = fronts[0]
            pareto_pop = [population[i] for i in pareto_indices]
            pareto_objs = objectives[pareto_indices]

            df_pareto = pd.DataFrame(pareto_objs, columns=self.obj_names)
            df_pareto['Prompt'] = pareto_pop
            save_path = f"{self.save_dir}/pareto_front_iter_{iteration:02d}.csv"
            df_pareto.to_csv(save_path, index=False, encoding='utf-8')
            return {'pareto_population': pareto_pop}
        except Exception:
            return None

    def save_metrics(self, metrics_dict, iteration):
        metrics_file = f"{self.save_dir}/metrics_log.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
        metrics_dict['iteration'] = iteration
        metrics_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data.append(metrics_dict)
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)

    def save_convergence_analysis(self, populations, objectives):
        pass

    def save_final_summary(self, final_pop, final_objs, total_time, final_metrics):
        try:
            fronts = nondomination.fast_non_dominated_sort(final_objs)
            pf_size = len(fronts[0]) if fronts else 0
        except:
            pf_size = 0

        summary = {
            'status': 'Completed',
            'total_time_sec': total_time,
            'final_metrics': final_metrics,
            'final_hv': self.history_records[-1]['Hypervolume'] if self.history_records else 0,
            'final_pareto_front_size': pf_size
        }
        with open(f"{self.save_dir}/final_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        self.create_readme()
        return summary

    def create_readme(self):
        readme = f"# 实验报告: {self.experiment_name}\n算法: {self.algorithm_name}\n目标: {self.obj_names}\n"
        with open(f"{self.save_dir}/README.md", 'w') as f:
            f.write(readme)

    def _plot_results(self, final_objs):
        pass