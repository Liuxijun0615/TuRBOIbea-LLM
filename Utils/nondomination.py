# Utils/nondomination.py
import numpy as np


def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points (Minimization logic)
    注意：此函数通常用于最小化问题（costs越小越好）。
    如果用于最大化问题，请先对输入取负或使用下面的 fast_non_dominated_sort。
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # And keep self
    return is_efficient


def fast_non_dominated_sort(objectives):
    """
    快速非支配排序算法 - NSGA-II 的核心算法
    **适用于最大化问题** (Maximization)

    输入: objectives - numpy数组，形状为(n, m)，n个解，m个目标函数
    输出: fronts - 列表，每个元素是一个前沿（解的索引列表）。fronts[0]是Pareto最优解。
    """
    # 安全检查
    if objectives is None or len(objectives) == 0:
        return []

    n = objectives.shape[0]
    if n == 0:
        return []

    # 初始化数据结构
    S = [[] for _ in range(n)]  # 被当前解支配的解集合 (Set of solutions dominated by p)
    n_p = np.zeros(n, dtype=int)  # 支配当前解的解的数量 (Domination count)
    rank = np.zeros(n, dtype=int)  # 每个解的排名 (Rank)
    fronts = [[]]  # 存储各前沿 (Fronts)

    # 第一遍：计算支配关系 O(N^2)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                # 判断 i 是否支配 j (最大化)
                if dominates(objectives[i], objectives[j]):
                    S[i].append(j)
                    n_p[j] += 1
                # 判断 j 是否支配 i (最大化)
                elif dominates(objectives[j], objectives[i]):
                    S[j].append(i)
                    n_p[i] += 1
            except Exception as e:
                print(f"⚠️ 支配关系计算出错: {e}")
                continue

        # 如果没有解支配 i，则 i 属于第一前沿 (Pareto Front)
        if n_p[i] == 0:
            rank[i] = 0
            fronts[0].append(i)

    # 如果第一前沿为空，直接返回
    if not fronts[0]:
        return []

    # 构建后续前沿
    i = 0
    while len(fronts) > i and fronts[i]:
        Q = []  # 下一前沿
        for p in fronts[i]:
            for q in S[p]:
                n_p[q] -= 1
                if n_p[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        if Q:
            fronts.append(Q)
        else:
            break

    return fronts


def dominates(a, b):
    """
    判断解 a 是否支配解 b (**最大化问题**)

    定义:
    1. a 在所有目标上都不比 b 差 (a >= b)
    2. a 至少在一个目标上严格优于 b (a > b)
    """
    # 检查 a 是否在所有目标上都不比 b 差
    not_worse = np.all(a >= b)
    # 检查 a 是否至少在一个目标上比 b 好
    better = np.any(a > b)

    return not_worse and better


def crowding_distance(objectives, front):
    """
    计算拥挤距离（用于NSGA-II的环境选择）
    输入: objectives - 目标函数值数组
          front - 前沿中的解索引列表
    输出: distances - 每个解的拥挤距离
    """
    n = len(front)
    distances = np.zeros(n)

    if n == 0:
        return distances

    m = objectives.shape[1]  # 目标函数数量

    for obj_idx in range(m):
        # 根据当前目标函数值排序前沿中的解
        sorted_indices = np.argsort(objectives[front, obj_idx])
        sorted_front = [front[i] for i in sorted_indices]

        # 边界解的拥挤距离设为无穷大
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        # 计算中间解的拥挤距离
        if n > 2:
            min_val = objectives[sorted_front[0], obj_idx]
            max_val = objectives[sorted_front[-1], obj_idx]

            # 避免除零
            if abs(max_val - min_val) < 1e-10:
                continue

            norm = max_val - min_val
            for i in range(1, n - 1):
                prev_val = objectives[sorted_front[i - 1], obj_idx]
                next_val = objectives[sorted_front[i + 1], obj_idx]
                distances[sorted_indices[i]] += (next_val - prev_val) / norm

    return distances


def get_pareto_front(objectives):
    """获取帕累托前沿（第一前沿）"""
    fronts = fast_non_dominated_sort(objectives)
    return fronts[0] if fronts else []


# ==========================================================
# 关键修改：添加函数别名以匹配 TuRBO_IBEA_LLM.py 的导入
# ==========================================================
non_domination_sort = fast_non_dominated_sort

# Test
if __name__ == "__main__":
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    print("开始测试非支配排序模块...")

    ########### 2D example ######################
    # 生成随机数据
    a = np.random.rand(20, 2)

    # 注意：is_pareto_efficient_simple 是最小化逻辑，而 fast_non_dominated_sort 是最大化逻辑
    # 为了对比，我们对 sort 的输入取负，模拟最小化，或者直接看最大化效果

    # 测试快速非支配排序 (最大化)
    fronts = non_domination_sort(a)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(a[:, 0], a[:, 1], 'bo', label='All Points')
    # 标记第一前沿
    if fronts:
        pf_points = a[fronts[0]]
        plt.plot(pf_points[:, 0], pf_points[:, 1], 'ro', label='Pareto Front (Max)')
    plt.title('Max Pareto Front (Rank 0)')
    plt.legend()

    plt.subplot(1, 2, 2)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, front in enumerate(fronts):
        if i < len(colors):
            color = colors[i]
        else:
            color = 'gray'
        front_points = a[front]
        plt.plot(front_points[:, 0], front_points[:, 1], 'o', color=color, label=f'Front {i}')
    plt.title('All Fronts (Maximization)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    ########### 3D example ######################
    a = np.random.rand(200, 3)
    fronts = non_domination_sort(a)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, front in enumerate(fronts[:3]):  # 只显示前3个前沿
        if i < len(colors):
            color = colors[i]
        else:
            color = 'gray'
        front_points = a[front]
        ax.scatter3D(front_points[:, 0], front_points[:, 1], front_points[:, 2], color=color, label=f'Front {i}')

    ax.legend()
    plt.title('3D Pareto Fronts (Maximization)')
    plt.show()

    print("测试完成。")