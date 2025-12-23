import matplotlib.pyplot as plt
import numpy as np
import colorsys

# 设置SCI论文风格
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 1.5

# 数据准备
problems = ['MoL-1M-1', 'MoL-1M-2', 'MoL-1M-3', 'Games-1', 'Games-2', 'Games-3',
            'Bundle-1', 'Bundle-2', 'Bundle-3']
algorithms = ['TuRBOIbea-LLM', 'LLM-NSGA-II', 'LLM-IBEA', 'LLM-MOEA/D']

# 各算法在不同问题上的耗时数据（秒）
time_data = np.array([
    [32862.32, 30356.39, 34328.43, 28387.67, 29798.35, 31983.85, 39798.42, 32648.76, 41243.98],  # TuRBOIbea-LLM
    [34762.94, 32656.76, 36383.92, 33998.72, 33984.41, 36387.67, 43489.58, 46983.37, 44845.35],  # LLM-NSGA-II
    [33853.78, 33387.43, 35987.32, 34198.48, 29887.37, 32573.29, 39384.25, 39736.64, 42887.49],  # LLM-IBEA
    [34876.28, 36948.21, 39938.57, 32098.73, 31897.43, 34589.21, 41098.73, 41746.98, 47588.67]  # LLM-MOEA/D
])

# 转换为小时单位便于阅读
time_data_hours = time_data / 3600

# 设置图形
fig, ax = plt.subplots(figsize=(12, 6))

# 设置位置参数
x = np.arange(len(problems))
width = 0.2


# 定义基础颜色并降低饱和度
def desaturate_color(hex_color, saturation_factor=0.7):
    """降低颜色饱和度"""
    # 将十六进制转换为RGB
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    # 转换为0-1范围
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # 转换为HSV并调整饱和度
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = s * saturation_factor  # 降低饱和度

    # 转换回RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    # 转换回十六进制
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


# 基础颜色
base_colors = ['#DE7833', '#F2BB6B', '#329845', '#276C9E']

# 降低饱和度后的颜色
colors = [desaturate_color(color, 1) for color in base_colors]

# 创建柱状图
bars1 = ax.bar(x - 1.5 * width, time_data_hours[0], width, label=algorithms[0],
               color=colors[0], edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x - 0.5 * width, time_data_hours[1], width, label=algorithms[1],
               color=colors[1], edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + 0.5 * width, time_data_hours[2], width, label=algorithms[2],
               color=colors[2], edgecolor='black', linewidth=0.5)
bars4 = ax.bar(x + 1.5 * width, time_data_hours[3], width, label=algorithms[3],
               color=colors[3], edgecolor='black', linewidth=0.5)

# 设置坐标轴标签
ax.set_xlabel('Problem Instances', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Execution Time (hours)', fontsize=11, fontweight='bold')

# 设置x轴刻度 - 水平标签
ax.set_xticks(x)
ax.set_xticklabels(problems, ha='center')  # 水平标签，居中对齐

# 设置y轴范围，留出空间显示数值标签
ax.set_ylim(0, max(time_data_hours.flatten()) * 1.15)

# 添加网格（浅灰色，仅y轴方向）
ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)  # 将网格线放在数据后面

# 添加图例
ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False,
          edgecolor='black', framealpha=0.9)


# 在柱子上添加数值标签（小时，保留1位小数）
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)


add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
add_value_labels(bars4)

# 移除上方和右方的边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 可选：保存为高分辨率图片（用于论文）
# plt.savefig('execution_time_comparison.png', dpi=300, bbox_inches='tight')
# plt.savefig('execution_time_comparison.pdf', bbox_inches='tight')