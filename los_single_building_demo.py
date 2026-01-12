"""单栋建筑 LoS 判定的演示脚本。"""

from __future__ import annotations

from typing import Iterable, Tuple

# 三维点：(x, y, z)
Point3 = Tuple[float, float, float]


def build_single_building(
    center_xy: Tuple[float, float] = (50.0, 0.0),
    size_xy: Tuple[float, float] = (20.0, 20.0),
    height: float = 30.0,
) -> Tuple[Point3, Point3]:
    """生成一栋建筑的 AABB（轴对齐包围盒）。"""
    cx, cy = center_xy
    sx, sy = size_xy
    hx = sx * 0.5
    hy = sy * 0.5
    return (cx - hx, cy - hy, 0.0), (cx + hx, cy + hy, height)


def segment_intersects_aabb(
    p0: Point3,
    p1: Point3,
    box_min: Point3,
    box_max: Point3,
    eps: float = 1e-9,
) -> bool:
    """判断线段是否与建筑物包围盒相交。"""
    t_min = 0.0
    t_max = 1.0
    # 线段方向向量。
    d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])

    for i in range(3):
        # 若线段几乎平行该轴，检查起点是否在范围内。
        if abs(d[i]) < eps:
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return False
            continue

        inv_d = 1.0 / d[i]
        # 计算与平面交点参数 t。
        t1 = (box_min[i] - p0[i]) * inv_d
        t2 = (box_max[i] - p0[i]) * inv_d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False

    return True


def is_los(uav: Point3, ground: Point3, box_min: Point3, box_max: Point3) -> bool:
    """线段未与建筑相交，则视为 LoS。"""
    return not segment_intersects_aabb(uav, ground, box_min, box_max)


def demo_cases() -> Iterable[Tuple[str, Point3, Point3]]:
    """准备几个可直观判断的测试场景。"""
    return [
        ("blocked", (0.0, 0.0, 20.0), (100.0, 0.0, 0.0)),
        ("clear_above", (0.0, 0.0, 80.0), (100.0, 0.0, 0.0)),
        ("clear_side", (0.0, 50.0, 20.0), (100.0, 50.0, 0.0)),
    ]


def main() -> None:
    """打印建筑物信息与每个测试场景的 LoS/NLoS。"""
    box_min, box_max = build_single_building()
    print("building_aabb:", box_min, box_max)

    for name, uav, ground in demo_cases():
        # 判断是否被建筑遮挡。
        los = is_los(uav, ground, box_min, box_max)
        status = "LoS" if los else "NLoS"
        print(f"{name:12s} uav={uav} ground={ground} -> {status}")


if __name__ == "__main__":
    # 直接运行该文件即可看到演示输出。
    main()
