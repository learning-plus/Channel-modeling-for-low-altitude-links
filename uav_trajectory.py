"""无人机位置/轨迹采样工具。"""

from __future__ import annotations

from typing import List, Tuple, Optional
import random

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


def generate_uav_positions_line(
    start: Point3,
    end: Point3,
    num_points: int,
) -> List[Point3]:
    """在线段上等间距采样若干个无人机位置。"""
    if num_points <= 1:
        return [start]
    positions: List[Point3] = []
    # 计算每一步的增量。
    step = (
        (end[0] - start[0]) / (num_points - 1),
        (end[1] - start[1]) / (num_points - 1),
        (end[2] - start[2]) / (num_points - 1),
    )
    for i in range(num_points):
        # 沿着线段逐点移动。
        positions.append(
            (start[0] + step[0] * i, start[1] + step[1] * i, start[2] + step[2] * i)
        )
    return positions


def generate_uav_positions_grid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    altitude: float,
    nx: int,
    ny: int,
) -> List[Point3]:
    """在矩形区域内生成规则网格的无人机位置。"""
    positions: List[Point3] = []
    if nx <= 0 or ny <= 0:
        return positions
    x_min, x_max = x_range
    y_min, y_max = y_range
    for ix in range(nx):
        # 在 x 方向均匀分布。
        x = x_min if nx == 1 else x_min + (x_max - x_min) * ix / (nx - 1)
        for iy in range(ny):
            # 在 y 方向均匀分布。
            y = y_min if ny == 1 else y_min + (y_max - y_min) * iy / (ny - 1)
            positions.append((x, y, altitude))
    return positions


def generate_uav_positions_random(
    area_min: Point2,
    area_max: Point2,
    altitude_range: Tuple[float, float],
    count: int,
    seed: Optional[int] = None,
) -> List[Point3]:
    """在给定区域内随机生成无人机位置。"""
    rng = random.Random(seed)
    x_min, y_min = area_min
    x_max, y_max = area_max
    z_min, z_max = altitude_range
    positions: List[Point3] = []
    for _ in range(count):
        # x、y、z 各自独立均匀抽样。
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        z = rng.uniform(z_min, z_max)
        positions.append((x, y, z))
    return positions
