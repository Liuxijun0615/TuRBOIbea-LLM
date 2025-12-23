import numpy as np


def calculate_hypervolume(points, reference_point):
    """
    计算超体积指标
    points: 帕累托前沿点，形状为(n_points, n_objectives)
    reference_point: 参考点，形状为(n_objectives,)
    """
    if len(points) == 0:
        return 0.0

    points = np.array(points)
    reference_point = np.array(reference_point)

    # 确保所有点都支配参考点
    if not np.all(points <= reference_point):
        # 如果有不支配参考点的点，调整参考点或过滤点
        valid_points = points[np.all(points <= reference_point, axis=1)]
        if len(valid_points) == 0:
            return 0.0
        points = valid_points

    # 对点进行排序
    sorted_points = points[np.lexsort(points.T)]

    # 计算超体积
    volume = 0.0
    prev_point = reference_point.copy()

    for point in sorted_points:
        # 计算当前点与前一点的体积
        segment_volume = np.prod(prev_point - point)
        volume += segment_volume
        prev_point = point

    return volume


def normalized_hypervolume(points, ideal_point, nadir_point):
    """
    计算归一化的超体积
    """
    if len(points) == 0:
        return 0.0

    # 归一化点
    normalized_points = (points - ideal_point) / (nadir_point - ideal_point)
    reference_point = np.ones(len(ideal_point))  # [1,1,...,1]

    return calculate_hypervolume(normalized_points, reference_point)


def hypervolume(objectives, ref_point):
    """
    兼容原有接口的超体积计算函数
    """
    try:
        if len(objectives) == 0:
            return 0.0

        # 对于最大化问题，参考点设为0，理想点设为1
        ref_point = np.zeros(objectives.shape[1])
        ideal_point = np.ones(objectives.shape[1])

        return normalized_hypervolume(objectives, ref_point, ideal_point)
    except Exception as e:
        print(f"⚠️ 超体积计算出错: {e}")
        return 0.0