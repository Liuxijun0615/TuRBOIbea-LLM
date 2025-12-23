import os
# 设置 Hugging Face 镜像地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from Problems import RCBench
from Algorithms.NSGA2_LLM import NSGA2_LLM
import json
import time
import pickle
import config

print("🚀 开始运行 NSGA2-LLM 实验")
print("=" * 60)

# Run Experiments
time_record = {}

for setting in config.DATA_OBJECTIVES:
    dataset, objectives = setting[0], setting[1]

    print(f"\n🎯 当前实验: {dataset} - {objectives}")
    print("-" * 40)

    # 设置优化任务
    func = eval(f'RCBench.{objectives}')

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
        "NSGA2-LLM", dataset, objectives, config.SEED
    )

    print(f"📁 详细结果: {detailed_dir}")
    print(f"📁 兼容路径: {compatibility_path}")

    # 进化优化
    print("🔄 开始进化优化...")
    start_time = time.time()

    try:
        Pop, Obj = NSGA2_LLM(
            problem=bench,
            max_iter=config.MAX_ITERATIONS,
            pop_size=config.POPULATION_SIZE,
            api_key=config.OPENAI_KEY,
            llm_model=config.LLM_MODEL,
            save_path=compatibility_path  # 使用兼容路径保持原有格式
        )
        end_time = time.time()

        experiment_time = end_time - start_time
        time_record[f"{dataset} & {objectives}"] = experiment_time

        print(f"✅ 实验完成 | 耗时: {experiment_time / 60:.2f}分钟")

    except Exception as e:
        print(f"❌ 实验失败: {e}")
        time_record[f"{dataset} & {objectives}"] = -1  # 标记失败

# 保存时间记录
time_file = f"{config.RESULTS_BASE_DIR}/TimeConsumption_NSGA2-LLM_Seed_{config.SEED}.pkl"
try:
    pickle.dump(time_record, open(time_file, "wb"))
    print(f"\n💾 时间记录已保存: {time_file}")
except Exception as e:
    print(f"❌ 时间记录保存失败: {e}")

print("\n" + "=" * 60)
print("🎉 NSGA2-LLM 所有实验完成!")
print(f"📊 成功实验: {sum(1 for t in time_record.values() if t > 0)}/{len(time_record)}")
print("=" * 60)