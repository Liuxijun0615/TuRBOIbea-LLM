# run_ablation.py
import os
# 设置 Hugging Face 镜像地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from Problems import RCBench
from Algorithms.TuRBO_IBEA_LLM_Ablation import TuRBO_IBEA_LLM_Ablation
import json
import time
import pickle
import config
import os
from datetime import datetime


def run_ablation_experiments():
    print("🚀 开始运行 TuRBO-IBEA-LLM 消融实验")
    print("=" * 60)

    # 定义要运行的消融模式
    ablation_modes = ['wo_mapping', 'wo_turbo']

    # 默认使用三目标设置进行消融实验
    # 你可以根据需要修改这里，例如遍历 config.DATA_OBJECTIVES
    target_settings = [
        ['Bundle', 'Acc_Div_Fair']
    ]

    for setting in target_settings:
        dataset, objectives = setting[0], setting[1]

        for mode in ablation_modes:
            mode_name = "w/o Mapping" if mode == 'wo_mapping' else "w/o TuRBO"
            print(f"\n🎯 当前消融实验: {dataset} - {objectives} [{mode_name}]")
            print("-" * 50)

            # 设置优化任务
            try:
                func = eval(f'RCBench.{objectives}')
            except AttributeError:
                print(f"❌ 找不到任务: RCBench.{objectives}")
                continue

            # 加载训练数据
            dataset_path = config.get_dataset_path(dataset, config.SEED, "train")
            try:
                with open(dataset_path, 'r', encoding='utf-8') as json_file:
                    train_data = json.load(json_file)
                print(f"✅ 数据加载成功: {dataset_path}")
            except Exception as e:
                print(f"❌ 数据加载失败: {e}")
                continue

            # 创建问题实例
            bench = func(
                train_data,
                config.BATCH_SIZE,
                config.OPENAI_KEY,
                llm_model=config.LLM_MODEL
            )

            # 生成区分消融模式的保存路径
            # 文件夹名示例: TuRBO_wo_Mapping_Acc_Div_Fair_...
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algo_folder_name = f"TuRBO_{mode}"

            # 兼容性路径 (用于传递给算法内部解析)
            compatibility_path = f"{config.RESULTS_BASE_DIR}/{dataset}/{algo_folder_name}_{objectives}_Seed_{config.SEED}"

            # 运行消融算法
            try:
                Pop, Obj = TuRBO_IBEA_LLM_Ablation(
                    problem=bench,
                    max_iter=config.MAX_ITERATIONS,
                    pop_size=config.POPULATION_SIZE,
                    api_key=config.OPENAI_KEY,
                    llm_model=config.LLM_MODEL,
                    save_path=compatibility_path,
                    ablation_mode=mode  # 传入当前模式
                )
                print(f"✅ {mode_name} 实验完成")

            except Exception as e:
                print(f"❌ {mode_name} 实验失败: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("🎉 所有消融实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_ablation_experiments()