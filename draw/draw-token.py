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

# 各算法在不同问题上的token消耗数据
token_data = np.array([
    [1924790, 1658656, 1896732, 1150908, 1486012, 1757972, 2897292, 3160960, 2853734],  # TuRBOIbea-LLM
    [2498202, 1970715, 2281927, 1824892, 2774380, 3158864, 3149372, 4357644, 3168362],  # LLM-NSGA-II
    [2100988, 2799094, 2177015, 1487300, 1539068, 2876542, 2950173, 3870493, 2953795],  # LLM-IBEA
    [2346827, 1977983, 1984762, 1250271, 1990173, 2680875, 3099349, 3372308, 3157300]   # LLM-MOEA/D
])

# 转换为百万单位便于阅读
token_data_millions = token_data / 1000000

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
bars1 = ax.bar(x - 1.5 * width, token_data_millions[0], width, label=algorithms[0],
               color=colors[0], edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x - 0.5 * width, token_data_millions[1], width, label=algorithms[1],
               color=colors[1], edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + 0.5 * width, token_data_millions[2], width, label=algorithms[2],
               color=colors[2], edgecolor='black', linewidth=0.5)
bars4 = ax.bar(x + 1.5 * width, token_data_millions[3], width, label=algorithms[3],
               color=colors[3], edgecolor='black', linewidth=0.5)

# 设置坐标轴标签
ax.set_xlabel('Problem Instances', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Token Consumption (millions)', fontsize=11, fontweight='bold')

# 设置x轴刻度 - 水平标签
ax.set_xticks(x)
ax.set_xticklabels(problems, ha='center')  # 水平标签，居中对齐

# 设置y轴范围，留出空间显示数值标签
ax.set_ylim(0, max(token_data_millions.flatten()) * 1.15)

# 添加网格（浅灰色，仅y轴方向）
ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)  # 将网格线放在数据后面

# 添加图例
ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False,
          edgecolor='black', framealpha=0.9)

# 在柱子上添加数值标签（百万单位，保留2位小数）
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                f'{height:.2f}',
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
# plt.savefig('token_consumption_comparison.png', dpi=300, bbox_inches='tight')
# plt.savefig('token_consumption_comparison.pdf', bbox_inches='tight')