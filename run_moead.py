# run_moead.py
from Problems import RCBench
from Algorithms.MOEAD_LLM import MOEAD_LLM
import time
import pickle
import json
import config
import numpy as np
import os


def run_moead_experiment():
    print("🚀 开始运行 MOEA/D-LLM 实验")
    print("=" * 60)

    # =======================================================
    # [关键调试代码] 强制打印配置参数，检查是否读取正确
    # =======================================================
    print(f"\n🔍 DEBUG: 配置参数检查")
    print(f"   - config文件路径: {os.path.abspath(config.__file__)}")
    print(f"   - POPULATION_SIZE: {config.POPULATION_SIZE}")
    print(f"   - NEIGHBORHOOD_SIZE (配置文件值): {config.NEIGHBORHOOD_SIZE}")

    # 强制检查逻辑：如果小于3，发出警告
    if config.NEIGHBORHOOD_SIZE < 3:
        print(f"⚠️ 警告: NEIGHBORHOOD_SIZE ({config.NEIGHBORHOOD_SIZE}) 过小！可能导致 IndexError。")
        print(f"⚠️ 建议: 请立即去 config.py 将其修改为 5。")
    else:
        print(f"✅ 参数检查通过: NEIGHBORHOOD_SIZE >= 3")
    print("=" * 60 + "\n")
    # =======================================================

    time_record = {}

    for setting in config.DATA_OBJECTIVES:
        dataset, objectives = setting[0], setting[1]

        print(f"\n🎯 当前实验: {dataset} - {objectives}")
        print("-" * 40)

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

        # 生成保存路径
        detailed_dir, compatibility_path = config.get_save_path(
            "MOEAD-LLM", dataset, objectives, config.SEED
        )

        print(f"📁 详细结果: {detailed_dir}")
        print(f"📁 兼容路径: {compatibility_path}")

        # 进化优化
        print("🔄 开始进化优化...")
        start_time = time.time()

        try:
            # =======================================================
            # [调试确认] 在调用前再次确认传入的参数
            # =======================================================
            print(f"⏳ 正在调用 MOEAD_LLM, num_sub_set={config.NEIGHBORHOOD_SIZE}...")

            Pop, Obj = MOEAD_LLM(
                problem=bench,
                max_iter=config.MAX_ITERATIONS,
                pop_size=config.POPULATION_SIZE,
                num_sub_set=config.NEIGHBORHOOD_SIZE,  # 确保传入的是 config 值
                api_key=config.OPENAI_KEY,
                llm_model=config.LLM_MODEL,
                save_path=compatibility_path
            )
            end_time = time.time()

            experiment_time = end_time - start_time
            time_record[f"{dataset} & {objectives}"] = experiment_time

            print(f"✅ 实验完成 | 耗时: {experiment_time / 60:.2f}分钟")
            print(f"📊 最终种群大小: {len(Pop)}")

        except Exception as e:
            print(f"❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            time_record[f"{dataset} & {objectives}"] = -1

    # 保存时间记录
    time_file = f"{config.RESULTS_BASE_DIR}/TimeConsumption_MOEAD-LLM_Seed_{config.SEED}.pkl"
    try:
        pickle.dump(time_record, open(time_file, "wb"))
        print(f"\n💾 时间记录已保存: {time_file}")
    except Exception as e:
        print(f"❌ 时间记录保存失败: {e}")

    print("\n" + "=" * 60)
    print("🎉 MOEA/D-LLM 所有实验结束")
    print("=" * 60)


if __name__ == "__main__":
    run_moead_experiment()