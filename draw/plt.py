import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import datetime


def load_pareto_data_from_csv(folder_path, algorithm_names=None):
    """
    从指定文件夹路径加载所有CSV文件中的帕累托前沿数据

    Parameters:
    folder_path (str): 包含CSV文件的文件夹路径

    Returns:
    dict: 包含算法名称和对应数据的字典
    """
    algorithm_data = {}

    # 获取文件夹中所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(folder_path, csv_file)

        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 提取目标函数值 (假设列名为Objective_1, Objective_2, Objective_3)
            objectives = df[['Objective_1', 'Objective_2', 'Objective_3']].values
            # 使用自定义算法名称或默认名称
            if algorithm_names and csv_file in algorithm_names:
                algorithm_name = algorithm_names[csv_file]
            else:
                # 使用文件名作为算法名称（去掉扩展名）
                algorithm_name = f"Algorithm {i + 1} ({os.path.splitext(csv_file)[0]})"

            algorithm_data[algorithm_name] = objectives

            print(f"成功加载 {csv_file}: {len(objectives)} 个解")

        except Exception as e:
            print(f"加载文件 {csv_file} 时出错: {e}")

    return algorithm_data


def save_plot_with_timestamp(fig, base_name, output_dir="plots"):
    """
    使用时间戳保存图片，避免覆盖

    Parameters:
    fig: matplotlib图形对象
    base_name (str): 图片基础名称
    output_dir (str): 输出目录
    """
    import datetime

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成带时间戳的文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # 保存图片
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图片已保存: {filepath}")


# 设置包含CSV文件的文件夹路径
folder_path = "/Users/lxj/PycharmProjects/OA/TuRBOIbea_LLM/Results/1-plt/bundle"  # 请替换为实际的文件夹路径
algorithm_names = {
    # 请根据您的实际文件名修改下面的映射
    "final_pareto_front.csv": "LLM-MOEAD",
    "final_pareto_front 2.csv": "LLM-IBEA",
    "final_pareto_front 3.csv": "TuRBOIbea-LLM",
    "final_pareto_front 4.csv": "LLM-NSGA-II",
}

# 加载数据
algorithm_data = load_pareto_data_from_csv(folder_path, algorithm_names)

if not algorithm_data:
    print("没有找到有效的CSV文件，请检查文件夹路径")
else:
    # 使用更美观的颜色方案 - 改为更鲜艳且易于区分的颜色
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33']
    markers = ['o', 's', '^', 'D', 'v', '<']

    # 设置全局字体和样式
    plt.rcParams.update({
        'font.size': 12,
        'font.weight': 'bold',  # 全局字体加粗
        'axes.labelweight': 'bold',  # 坐标轴标签加粗
        'axes.titleweight': 'bold',  # 标题加粗
    })

    # 创建图形
    fig = plt.figure(figsize=(15, 5))

    # 方法1: 3D散点图
    ax1 = fig.add_subplot(131, projection='3d')

    # 绘制每个算法的帕累托前沿点（无连线）
    for i, (algo_name, data) in enumerate(algorithm_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax1.scatter(data[:, 0], data[:, 1], data[:, 2],
                    c=color, marker=marker, s=100, label=algo_name, alpha=0.8,
                    edgecolors='black', linewidth=0.5)  # 添加黑色边框使点更清晰

    ax1.set_xlabel('Acc', fontweight='bold')  # 坐标轴标签加粗
    ax1.set_ylabel('Div', fontweight='bold')
    ax1.set_zlabel('Fair', fontweight='bold')
    ax1.set_title('3D Pareto Front Comparison', fontweight='bold', fontsize=14)  # 标题加粗并增大字号
    ax1.legend()
    # 设置3D图坐标轴范围为0-1
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    # 设置更好的视角
    ax1.view_init(elev=20, azim=45)

    # 方法2: 三个2D投影图
    # Objective 1 vs Objective 2
    ax2 = fig.add_subplot(132)
    for i, (algo_name, data) in enumerate(algorithm_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax2.scatter(data[:, 0], data[:, 1], c=color, marker=marker, s=80,
                    label=algo_name, alpha=0.8, edgecolors='black', linewidth=0.5)

    ax2.set_xlabel('Acc', fontweight='bold')
    ax2.set_ylabel('Div', fontweight='bold')
    ax2.set_title('Acc vs Div', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # 设置坐标轴范围为0-1
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Objective 1 vs Objective 3
    ax3 = fig.add_subplot(133)
    for i, (algo_name, data) in enumerate(algorithm_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax3.scatter(data[:, 0], data[:, 2], c=color, marker=marker, s=80,
                    label=algo_name, alpha=0.8, edgecolors='black', linewidth=0.5)

    ax3.set_xlabel('Acc', fontweight='bold')
    ax3.set_ylabel('Fair', fontweight='bold')
    ax3.set_title('Acc vs Fair', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # 设置坐标轴范围为0-1
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    plt.tight_layout()

    # 保存第一个图形
    save_plot_with_timestamp(fig, "pareto_front_comparison_1")

    plt.show()

    # 创建另一个图形来显示Objective 2 vs Objective 3
    fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))

    # Objective 2 vs Objective 3
    for i, (algo_name, data) in enumerate(algorithm_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax4.scatter(data[:, 1], data[:, 2], c=color, marker=marker, s=80,
                    label=algo_name, alpha=0.8, edgecolors='black', linewidth=0.5)

    ax4.set_xlabel('Div', fontweight='bold')
    ax4.set_ylabel('Fair', fontweight='bold')
    ax4.set_title('Div vs Fair', fontweight='bold', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    # 设置坐标轴范围为0-1
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    # 平行坐标图
    objectives_names = ['Acc', 'Div', 'Fair']  # 更新为更直观的标签
    for i, (algo_name, data) in enumerate(algorithm_data.items()):
        color = colors[i % len(colors)]

        # 绘制每个解
        for j, point in enumerate(data):
            # 只绘制前几个点，避免图例过于拥挤
            if j < 3:  # 每个算法最多显示3个点在图例中
                label = f'{algo_name} - Point {j + 1}' if j == 0 else ""
            else:
                label = ""

            ax5.plot([0, 1, 2], point, c=color, marker=markers[i % len(markers)],
                     linewidth=1.5, markersize=6, label=label, alpha=0.7)

    ax5.set_xlabel('Objectives', fontweight='bold')
    ax5.set_ylabel('Values', fontweight='bold')
    ax5.set_title('Parallel Coordinates Plot', fontweight='bold', fontsize=14)
    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(objectives_names)
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    # 设置平行坐标图的y轴范围为0-1
    ax5.set_ylim(0, 1)

    plt.tight_layout()

    # 保存第二个图形
    save_plot_with_timestamp(fig2, "pareto_front_comparison_2")

    plt.show()

    # 打印统计信息
    print("\n帕累托前沿统计信息:")
    print("=" * 50)
    for algo_name, data in algorithm_data.items():
        print(f"{algo_name}:")
        print(f"  解的数量: {len(data)}")
        print(f"  Objective 1 范围: [{data[:, 0].min():.6f}, {data[:, 0].max():.6f}]")
        print(f"  Objective 2 范围: [{data[:, 1].min():.6f}, {data[:, 1].max():.6f}]")
        print(f"  Objective 3 范围: [{data[:, 2].min():.6f}, {data[:, 2].max():.6f}]")
        print()

    # 将统计信息保存到文件
    stats_filename = f"pareto_stats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    stats_filepath = os.path.join("plots", stats_filename)

    with open(stats_filepath, 'w') as f:
        f.write("帕累托前沿统计信息:\n")
        f.write("=" * 50 + "\n")
        for algo_name, data in algorithm_data.items():
            f.write(f"{algo_name}:\n")
            f.write(f"  解的数量: {len(data)}\n")
            f.write(f"  Objective 1 范围: [{data[:, 0].min():.6f}, {data[:, 0].max():.6f}]\n")
            f.write(f"  Objective 2 范围: [{data[:, 1].min():.6f}, {data[:, 1].max():.6f}]\n")
            f.write(f"  Objective 3 范围: [{data[:, 2].min():.6f}, {data[:, 2].max():.6f}]\n")
            f.write("\n")

    print(f"统计信息已保存: {stats_filepath}")