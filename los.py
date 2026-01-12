"""LoS/NLoS 判定工具：判断连线是否被建筑物遮挡。"""

from __future__ import annotations

from typing import Iterable, Tuple

from city_env import Building, Point3


def segment_intersects_aabb(
    p0: Point3,
    p1: Point3,
    box_min: Point3,
    box_max: Point3,
    eps: float = 1e-9,
) -> bool:
    """判断线段 p0->p1 是否与轴对齐包围盒相交。"""
    # t 在 [0, 1] 表示在线段上移动的比例。
    t_min = 0.0
    t_max = 1.0
    # 线段方向向量。
    d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])

    for i in range(3):
        # 近似平行某轴时，只能在该轴范围内才可能相交。
        if abs(d[i]) < eps:
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return False
            continue

        inv_d = 1.0 / d[i]
        # 计算该轴上与两个平面的交点参数。
        t1 = (box_min[i] - p0[i]) * inv_d
        t2 = (box_max[i] - p0[i]) * inv_d
        if t1 > t2:
            t1, t2 = t2, t1
        # 将该轴的范围与全局范围求交集。
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False

    return True


def is_los(uav: Point3, ground: Point3, buildings: Iterable[Building]) -> bool:
    """只要线段与任一建筑相交，就视为 NLoS。"""
    for building in buildings:
        box_min, box_max = building.aabb()
        if segment_intersects_aabb(uav, ground, box_min, box_max):
            return False
    return True
