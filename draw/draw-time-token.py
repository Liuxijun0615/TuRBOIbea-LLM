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

# 执行时间数据（秒）
time_data = np.array([
    [32862.32, 30356.39, 34328.43, 28387.67, 29798.35, 31983.85, 39798.42, 32648.76, 41243.98],  # TuRBOIbea-LLM
    [34762.94, 32656.76, 36383.92, 33998.72, 33984.41, 36387.67, 43489.58, 46983.37, 44845.35],  # LLM-NSGA-II
    [33853.78, 33387.43, 35987.32, 34198.48, 29887.37, 32573.29, 39384.25, 39736.64, 42887.49],  # LLM-IBEA
    [34876.28, 36948.21, 39938.57, 32098.73, 31897.43, 34589.21, 41098.73, 41746.98, 47588.67]  # LLM-MOEA/D
])

# Token消耗数据
token_data = np.array([
    [1924790, 1658656, 1896732, 1150908, 1486012, 1757972, 2897292, 3160960, 2853734],  # TuRBOIbea-LLM
    [2498202, 1970715, 2281927, 1824892, 2774380, 3158864, 3149372, 4357644, 3168362],  # LLM-NSGA-II
    [2100988, 2799094, 2177015, 1487300, 1539068, 2876542, 2950173, 3870493, 2953795],  # LLM-IBEA
    [2346827, 1977983, 1984762, 1250271, 1990173, 2680875, 3099349, 3372308, 3157300]   # LLM-MOEA/D
])

# 转换单位
time_data_hours = time_data / 3600
token_data_millions = token_data / 1000000

# 定义基础颜色并降低饱和度
def desaturate_color(hex_color, saturation_factor=0.7):
    """降低颜色饱和度"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = s * saturation_factor
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))

# 基础颜色
base_colors = ['#DE7833', '#F2BB6B', '#329845', '#276C9E']
colors = [desaturate_color(color, 1) for color in base_colors]

# 创建包含两个子图的图形（横向排布），增加横向间距
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.3})

# 设置位置参数
x = np.arange(len(problems))
width = 0.2

# === 子图a：执行时间 ===
# 创建柱状图
bars1_a = ax1.bar(x - 1.5 * width, time_data_hours[0], width, label=algorithms[0],
                  color=colors[0], edgecolor='black', linewidth=0.5)
bars2_a = ax1.bar(x - 0.5 * width, time_data_hours[1], width, label=algorithms[1],
                  color=colors[1], edgecolor='black', linewidth=0.5)
bars3_a = ax1.bar(x + 0.5 * width, time_data_hours[2], width, label=algorithms[2],
                  color=colors[2], edgecolor='black', linewidth=0.5)
bars4_a = ax1.bar(x + 1.5 * width, time_data_hours[3], width, label=algorithms[3],
                  color=colors[3], edgecolor='black', linewidth=0.5)

# 设置坐标轴标签
ax1.set_xlabel('Problem Instances', fontsize=11, fontweight='bold')
ax1.set_ylabel('Average Execution Time (hours)', fontsize=11, fontweight='bold')

# 设置x轴刻度 - 增加间距并水平放置
ax1.set_xticks(x)
ax1.set_xticklabels(problems, ha='center', fontsize=9)  # 减小字体大小以增加间距感

# 设置y轴范围
ax1.set_ylim(0, max(time_data_hours.flatten()) * 1.15)

# 添加网格
ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# 移除边框线
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 添加子图标题
ax1.set_title('(a) Average time consumption (hours)', fontsize=12, fontweight='bold', pad=20)

# 添加图例到子图a
ax1.legend(loc='upper left', frameon=True, fancybox=False, shadow=False,
          edgecolor='black', framealpha=0.9, fontsize=9)

# === 子图b：Token消耗 ===
# 创建柱状图
bars1_b = ax2.bar(x - 1.5 * width, token_data_millions[0], width, label=algorithms[0],
                  color=colors[0], edgecolor='black', linewidth=0.5)
bars2_b = ax2.bar(x - 0.5 * width, token_data_millions[1], width, label=algorithms[1],
                  color=colors[1], edgecolor='black', linewidth=0.5)
bars3_b = ax2.bar(x + 0.5 * width, token_data_millions[2], width, label=algorithms[2],
                  color=colors[2], edgecolor='black', linewidth=0.5)
bars4_b = ax2.bar(x + 1.5 * width, token_data_millions[3], width, label=algorithms[3],
                  color=colors[3], edgecolor='black', linewidth=0.5)

# 设置坐标轴标签
ax2.set_xlabel('Problem Instances', fontsize=11, fontweight='bold')
ax2.set_ylabel('Average Token Consumption (millions)', fontsize=11, fontweight='bold')

# 设置x轴刻度 - 增加间距并水平放置
ax2.set_xticks(x)
ax2.set_xticklabels(problems, ha='center', fontsize=9)  # 减小字体大小以增加间距感

# 设置y轴范围
ax2.set_ylim(0, max(token_data_millions.flatten()) * 1.15)

# 添加网格
ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)

# 移除边框线
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 添加子图标题
ax2.set_title('(b) Average Token Consumption', fontsize=12, fontweight='bold', pad=20)

# 添加图例到子图b
ax2.legend(loc='upper left', frameon=True, fancybox=False, shadow=False,
          edgecolor='black', framealpha=0.9, fontsize=9)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 保存为高分辨率图片（用于论文）
# plt.savefig('combined_comparison.png', dpi=300, bbox_inches='tight')
# plt.savefig('combined_comparison.pdf', bbox_inches='tight')
