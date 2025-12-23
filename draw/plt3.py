import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import datetime
import seaborn as sns  # 添加seaborn用于更好的颜色风格


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

            # 使用文件名作为算法名称（去掉扩展名）
            #  algorithm_name = f"Algorithm {i + 1} ({os.path.splitext(csv_file)[0]})"
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


# 设置matplotlib样式和全局字体
plt.style.use('seaborn-v0_8-whitegrid')  # 使用seaborn白色网格风格

# 设置全局字体和坐标轴粗细 - 修改字体为更专业的字体
plt.rcParams['font.family'] = 'DejaVu Sans'  # 专业美观的无衬线字体
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.5  # 坐标轴线宽

# 设置包含CSV文件的文件夹路径
folder_path = "/Users/lxj/PycharmProjects/OA/TuRBOIbea_LLM/Results/1-plt/game"  # 请替换为实际的文件夹路径
# algorithm_names = {
#     # 请根据您的实际文件名修改下面的映射
#     "final_pareto_front.csv": "LLM-MOEAD",
#     "final_pareto_front 2.csv": "LLM-IBEA",
#     "final_pareto_front 3.csv": "TuRBOIbea-LLM",
#     "final_pareto_front 4.csv": "LLM-NSGA-II",
# }
algorithm_names = {
    # 请根据您的实际文件名修改下面的映射
    "moead.csv": "LLM-MOEAD",
    "ibea.csv": "LLM-IBEA",
    "turbo.csv": "TuRBOIbea-LLM",
    "nsga.csv": "LLM-NSGA-II",
}
# 加载数据
# algorithm_data = load_pareto_data_from_csv(folder_path)
algorithm_data = load_pareto_data_from_csv(folder_path, algorithm_names)

if not algorithm_data:
    print("没有找到有效的CSV文件，请检查文件夹路径")
else:
    # 定义更柔和的颜色和标记 - 使用seaborn调色板
    colors = sns.color_palette("husl", len(algorithm_data))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # 创建图形
    fig = plt.figure(figsize=(15, 5))

    # 方法1: 3D散点图
    ax1 = fig.add_subplot(131, projection='3d')

    # 绘制每个算法的帕累托前沿点（无连线）- 减小点大小
    for i, (algo_name, data) in enumerate(algorithm_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax1.scatter(data[:, 0], data[:, 1], data[:, 2],
                    c=[color], marker=marker, s=40, label=algo_name, alpha=0.7,
                    edgecolors='w', linewidth=0.5)  # 添加白色边框

    # 加粗3D图的坐标轴标签和标题 - 使用更专业的字体
    ax1.set_xlabel('Acc', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax1.set_ylabel('Div', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax1.set_zlabel('Fair', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax1.set_title('3D Pareto Front Comparison', fontweight='bold', fontsize=14, fontfamily='DejaVu Sans')

    # 加粗3D图的坐标轴线
    ax1.xaxis.line.set_linewidth(1.5)
    ax1.yaxis.line.set_linewidth(1.5)
    ax1.zaxis.line.set_linewidth(1.5)

    # 加粗刻度标签 - 使用更专业的字体
    ax1.tick_params(axis='both', which='major', width=1.5, labelsize=10, labelfamily='DejaVu Sans')

    ax1.legend(fontsize=9, prop={'family': 'DejaVu Sans'})
    # 设置3D图坐标轴范围为0-1
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)

    # 改进：设置更好的视角
    ax1.view_init(elev=20, azim=45)  # 仰角20度，方位角45度

    # 改进：添加网格以便更好地区分坐标轴
    ax1.grid(True, alpha=0.3)

    # 方法2: 三个2D投影图
    # Objective 1 vs Objective 2
    ax2 = fig.add_subplot(132)
    for i, (algo_name, data) in enumerate(algorithm_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax2.scatter(data[:, 0], data[:, 1], c=[color], marker=marker, s=30,
                    label=algo_name, alpha=0.7, edgecolors='w', linewidth=0.3)

    # 加粗2D图的坐标轴标签和标题 - 使用更专业的字体
    ax2.set_xlabel('Acc', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax2.set_ylabel('Div', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax2.set_title('Acc vs Div', fontweight='bold', fontsize=14, fontfamily='DejaVu Sans')

    # 加粗坐标轴线
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    # 加粗刻度 - 使用更专业的字体
    ax2.tick_params(axis='both', which='major', width=1.5, labelsize=10, labelfamily='DejaVu Sans')

    ax2.legend(fontsize=9, prop={'family': 'DejaVu Sans'})
    ax2.grid(True, alpha=0.3)
    # 设置坐标轴范围为0-1
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Objective 1 vs Objective 3
    ax3 = fig.add_subplot(133)
    for i, (algo_name, data) in enumerate(algorithm_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax3.scatter(data[:, 0], data[:, 2], c=[color], marker=marker, s=30,
                    label=algo_name, alpha=0.7, edgecolors='w', linewidth=0.3)

    # 加粗2D图的坐标轴标签和标题 - 使用更专业的字体
    ax3.set_xlabel('Acc', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax3.set_ylabel('Fair', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax3.set_title('Acc vs Fair', fontweight='bold', fontsize=14, fontfamily='DejaVu Sans')

    # 加粗坐标轴线
    for spine in ax3.spines.values():
        spine.set_linewidth(1.5)

    # 加粗刻度 - 使用更专业的字体
    ax3.tick_params(axis='both', which='major', width=1.5, labelsize=10, labelfamily='DejaVu Sans')

    ax3.legend(fontsize=9, prop={'family': 'DejaVu Sans'})
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

        ax4.scatter(data[:, 1], data[:, 2], c=[color], marker=marker, s=30,
                    label=algo_name, alpha=0.7, edgecolors='w', linewidth=0.3)

    # 加粗2D图的坐标轴标签和标题 - 使用更专业的字体
    ax4.set_xlabel('Div', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax4.set_ylabel('Fair', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax4.set_title('Div vs Fair', fontweight='bold', fontsize=14, fontfamily='DejaVu Sans')

    # 加粗坐标轴线
    for spine in ax4.spines.values():
        spine.set_linewidth(1.5)

    # 加粗刻度 - 使用更专业的字体
    ax4.tick_params(axis='both', which='major', width=1.5, labelsize=10, labelfamily='DejaVu Sans')

    ax4.legend(fontsize=9, prop={'family': 'DejaVu Sans'})
    ax4.grid(True, alpha=0.3)
    # 设置坐标轴范围为0-1
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    # 平行坐标图 - 使用更细的线条和更小的点
    objectives_names = ['Acc', 'Div', 'Fair']  # 使用实际的目标名称
    for i, (algo_name, data) in enumerate(algorithm_data.items()):
        color = colors[i % len(colors)]

        # 绘制每个解
        for j, point in enumerate(data):
            # 只绘制前几个点，避免图例过于拥挤
            if j < 3:  # 每个算法最多显示3个点在图例中
                label = f'{algo_name}' if j == 0 else ""  # 简化图例标签
            else:
                label = ""

            ax5.plot([0, 1, 2], point, c=color, marker=markers[i % len(markers)],
                     linewidth=0.5, markersize=3, label=label, alpha=0.6)  # 减小线宽和点大小

    # 加粗平行坐标图的坐标轴标签和标题 - 使用更专业的字体
    ax5.set_xlabel('Objectives', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax5.set_ylabel('Values', fontweight='bold', fontsize=12, fontfamily='DejaVu Sans')
    ax5.set_title('Parallel Coordinates Plot', fontweight='bold', fontsize=14, fontfamily='DejaVu Sans')

    # 加粗坐标轴线
    for spine in ax5.spines.values():
        spine.set_linewidth(1.5)

    # 加粗刻度 - 使用更专业的字体
    ax5.tick_params(axis='both', which='major', width=1.5, labelsize=10, labelfamily='DejaVu Sans')

    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(objectives_names, fontweight='bold', fontfamily='DejaVu Sans')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, prop={'family': 'DejaVu Sans'})
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