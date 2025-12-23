import numpy as np
from scipy.spatial.distance import cdist


def extract_edit_prompt(response):
    import re
    # 增加更多匹配模式，提高提取成功率
    patterns = [
        r'<START>\s*(.*?)\s*<END>',
        r'```(?:python)?\s*(.*?)\s*```',
        r'["\']([^"\']*?)["\']',
        r'Prompt:\s*(.*?)(?:\n\n|$)',
        r'Revised prompt:\s*(.*?)(?:\n\n|$)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            cleaned = [match.strip() for match in matches if match.strip()]
            if cleaned:
                return cleaned

    # 如果正则提取失败，尝试简单的行过滤
    lines = response.split('\n')
    meaningful_lines = [line.strip() for line in lines
                        if line.strip() and len(line.strip()) > 10
                        and not line.strip().startswith(('Here', 'Sure', 'I can', 'The prompt'))]
    if meaningful_lines:
        return [meaningful_lines[0]]

    return [response.strip()] if response.strip() else []


def environment_selection(population, n_survive):
    pop = population[0]
    objs = population[1]

    # 非支配排序
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds = NonDominatedSorting()
    fronts = nds.do(objs)

    # 选择
    next_pop = []
    next_objs = []

    for front in fronts:
        if len(next_pop) + len(front) <= n_survive:
            for idx in front:
                next_pop.append(pop[idx])
                next_objs.append(objs[idx])
        else:
            # 拥挤距离排序
            remaining = n_survive - len(next_pop)
            if remaining > 0:
                # 计算当前层的拥挤距离
                current_front_objs = objs[front]
                crowding_dist = calc_crowding_distance(current_front_objs)

                # 根据拥挤距离降序排列
                sorted_indices = np.argsort(crowding_dist)[::-1]
                selected_indices = [front[i] for i in sorted_indices[:remaining]]

                for idx in selected_indices:
                    next_pop.append(pop[idx])
                    next_objs.append(objs[idx])
            break

    return [next_pop, np.array(next_objs)], None, None, None


def calc_crowding_distance(f):
    n_points, n_obj = f.shape
    dist = np.zeros(n_points)

    for m in range(n_obj):
        # 对该目标进行排序
        sorted_indices = np.argsort(f[:, m])
        obj_min = f[sorted_indices[0], m]
        obj_max = f[sorted_indices[-1], m]

        # 边界点距离设为无穷大
        dist[sorted_indices[0]] = np.inf
        dist[sorted_indices[-1]] = np.inf

        # 防止除零
        denominator = obj_max - obj_min
        if denominator == 0: denominator = 1e-9

        for i in range(1, n_points - 1):
            dist[sorted_indices[i]] += (f[sorted_indices[i + 1], m] - f[sorted_indices[i - 1], m]) / denominator

    return dist


def Tchebycheff(pop_obj, weights):
    # 归一化
    f_min = np.min(pop_obj, axis=0)
    f_max = np.max(pop_obj, axis=0)

    # 防止微小差异被错误放大导致 HV=1.0
    denominator = np.maximum(f_max - f_min, 1e-2)

    normalized_obj = (pop_obj - f_min) / denominator

    if normalized_obj.ndim == 1:
        return np.max(normalized_obj * weights)
    else:
        return np.max(normalized_obj * weights, axis=1)


def choice_matrix(p_sel, n_per):
    """
    从邻域中选择 n_per 个索引
    """
    pop_size, n_neighbor = p_sel.shape
    idx_sel = np.zeros([n_per, pop_size], dtype=int)

    for i in range(pop_size):
        # [关键修复]
        # 如果邻域大小(n_neighbor) 小于 需要选取的数量(n_per)
        # 必须允许重复采样 (replace=True)，否则会报 ValueError
        use_replace = False
        if n_neighbor < n_per:
            use_replace = True

        idx_sel[:, i] = np.random.choice(n_neighbor, n_per, replace=use_replace, p=p_sel[i, :])

    return idx_sel


def IBEA_Selection(pop, y_pop, n_survive, kappa=0.05):
    # IBEA 环境选择实现
    n_pop = len(pop)
    if n_pop <= n_survive:
        return pop, y_pop

    # 归一化
    f_min = np.min(y_pop, axis=0)
    f_max = np.max(y_pop, axis=0)

    denominator = np.maximum(f_max - f_min, 1e-2)
    scaled_y_pop = (y_pop - f_min) / denominator

    # 计算适应度
    fitness = np.zeros(n_pop)
    for i in range(n_pop):
        for j in range(n_pop):
            if i != j:
                max_diff = np.max(scaled_y_pop[j] - scaled_y_pop[i])
                fitness[i] += -np.exp(-max_diff / kappa)

    # 迭代移除最差个体
    indices = list(range(n_pop))
    while len(indices) > n_survive:
        min_fitness_idx = np.argmin(fitness[indices])
        remove_idx = indices[min_fitness_idx]

        # 更新剩余个体适应度
        for i in indices:
            if i != remove_idx:
                max_diff = np.max(scaled_y_pop[remove_idx] - scaled_y_pop[i])
                fitness[i] -= -np.exp(-max_diff / kappa)

        indices.pop(min_fitness_idx)

    return [pop[i] for i in indices], y_pop[indices]