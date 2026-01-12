# -*- coding: utf-8 -*-
"""城市环境与建筑物生成的简单工具模块。

这个文件只负责“几何场景”的构建，不涉及信号模型。
核心输出是一组矩形建筑（中心点、尺寸、高度），用于后续 LoS 判定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

# 二维点：(x, y)
Point2 = Tuple[float, float]
# 三维点：(x, y, z)
Point3 = Tuple[float, float, float]
# 高度分布配置：(权重, (最小高度, 最大高度))
HeightTier = Tuple[float, Tuple[float, float]]


@dataclass(frozen=True)
class Building:
    """单个矩形建筑物，用中心点 + 尺寸 + 高度描述。"""

    center_xy: Point2
    size_xy: Point2
    height: float

    def aabb(self) -> Tuple[Point3, Point3]:
        """返回建筑物的轴对齐包围盒(AABB)两端点。"""
        cx, cy = self.center_xy
        sx, sy = self.size_xy
        hx = sx * 0.5
        hy = sy * 0.5
        # AABB 的最小角点与最大角点。
        return (cx - hx, cy - hy, 0.0), (cx + hx, cy + hy, self.height)


def _rects_overlap(
    a_min: Point2,
    a_max: Point2,
    b_min: Point2,
    b_max: Point2,
    gap: float,
) -> bool:
    """判断两个矩形(地面投影)是否重叠,gap 代表最小间距。"""
    # 若在 x 方向完全分离，则不重叠。
    if a_max[0] + gap <= b_min[0] or b_max[0] + gap <= a_min[0]:
        return False
    # 若在 y 方向完全分离，则不重叠。
    if a_max[1] + gap <= b_min[1] or b_max[1] + gap <= a_min[1]:
        return False
    return True


def _sample_height(rng: random.Random, profile: List[HeightTier]) -> float:
    """按高度分布配置抽样建筑高度（权重越大越容易被抽到）。"""
    total = sum(weight for weight, _ in profile)
    if total <= 0:
        raise ValueError("Height profile weights must be positive.")
    r = rng.uniform(0.0, total)
    acc = 0.0
    for weight, (h_min, h_max) in profile:
        acc += weight
        if r <= acc:
            return rng.uniform(h_min, h_max)
    # 理论上不会走到这里，保险起见给个默认值。
    return rng.uniform(profile[-1][1][0], profile[-1][1][1])


def _sample_size(rng: random.Random, size_range: Tuple[float, float], size_skew: float) -> float:
    """按 size_skew 偏向抽样尺寸，>1 更偏向小楼。"""
    size_min, size_max = size_range
    if size_skew <= 0:
        raise ValueError("size_skew must be positive.")
    # 用幂次来“压缩”随机数，让小尺寸更容易出现。
    u = rng.random() ** size_skew
    return size_min + (size_max - size_min) * u


def _make_block_ranges(min_val: float, max_val: float, block_size: float, road_width: float) -> List[Tuple[float, float]]:
    """把一条轴划分为“街区 + 道路”的连续区间（只返回街区区间）。"""
    ranges: List[Tuple[float, float]] = []
    cursor = min_val
    while cursor < max_val:
        block_start = cursor
        block_end = min(cursor + block_size, max_val)
        if block_end <= block_start:
            break
        ranges.append((block_start, block_end))
        cursor = block_end + road_width
    return ranges


def _generate_buildings_density(
    area_min: Point2,
    area_max: Point2,
    density: float,
    size_range: Tuple[float, float],
    height_profile: List[HeightTier],
    size_skew: float,
    min_gap: float,
    max_tries: int,
    rng: random.Random,
) -> List[Building]:
    """在指定矩形区域内，按覆盖率密度生成建筑物列表。"""
    x_min, y_min = area_min
    x_max, y_max = area_max
    size_min, size_max = size_range

    area_total = (x_max - x_min) * (y_max - y_min)
    target_area = density * area_total
    buildings: List[Building] = []
    used_area = 0.0
    tries = 0

    while used_area < target_area and tries < max_tries:
        tries += 1
        sx = _sample_size(rng, size_range, size_skew)
        sy = _sample_size(rng, size_range, size_skew)
        hx = sx * 0.5
        hy = sy * 0.5
        if x_max - x_min < sx or y_max - y_min < sy:
            break

        cx = rng.uniform(x_min + hx, x_max - hx)
        cy = rng.uniform(y_min + hy, y_max - hy)
        height = _sample_height(rng, height_profile)
        new_building = Building(center_xy=(cx, cy), size_xy=(sx, sy), height=height)

        a_min, a_max = new_building.aabb()
        a_min2 = (a_min[0], a_min[1])
        a_max2 = (a_max[0], a_max[1])

        overlap = False
        for building in buildings:
            b_min, b_max = building.aabb()
            b_min2 = (b_min[0], b_min[1])
            b_max2 = (b_max[0], b_max[1])
            if _rects_overlap(a_min2, a_max2, b_min2, b_max2, min_gap):
                overlap = True
                break

        if not overlap:
            buildings.append(new_building)
            used_area += sx * sy

    return buildings


def generate_buildings(
    area_min: Point2,
    area_max: Point2,
    count: int,
    size_range: Tuple[float, float],
    height_range: Tuple[float, float],
    min_gap: float = 0.0,
    max_tries: int = 2000,
    seed: Optional[int] = None,
) -> List[Building]:
    """按指定数量生成建筑物（简单版本，用 count 控制数量）。"""
    rng = random.Random(seed)
    x_min, y_min = area_min
    x_max, y_max = area_max
    size_min, size_max = size_range
    h_min, h_max = height_range

    if size_min <= 0 or size_max <= 0 or h_min <= 0 or h_max <= 0:
        raise ValueError("size_range and height_range must be positive.")

    buildings: List[Building] = []
    tries = 0
    # 不断尝试随机放置建筑，直到数量够或尝试次数耗尽。
    while len(buildings) < count and tries < max_tries:
        tries += 1
        sx = rng.uniform(size_min, size_max)
        sy = rng.uniform(size_min, size_max)
        hx = sx * 0.5
        hy = sy * 0.5
        # 如果建筑比区域还大，直接停止。
        if x_max - x_min < sx or y_max - y_min < sy:
            break

        # 在区域内随机一个中心点。
        cx = rng.uniform(x_min + hx, x_max - hx)
        cy = rng.uniform(y_min + hy, y_max - hy)
        height = rng.uniform(h_min, h_max)
        new_building = Building(center_xy=(cx, cy), size_xy=(sx, sy), height=height)

        a_min, a_max = new_building.aabb()
        a_min2 = (a_min[0], a_min[1])
        a_max2 = (a_max[0], a_max[1])

        overlap = False
        # 逐个检查与已放置建筑是否重叠。
        for building in buildings:
            b_min, b_max = building.aabb()
            b_min2 = (b_min[0], b_min[1])
            b_max2 = (b_max[0], b_max[1])
            if _rects_overlap(a_min2, a_max2, b_min2, b_max2, min_gap):
                overlap = True
                break

        if not overlap:
            buildings.append(new_building)

    return buildings


def generate_city_buildings(
    area_min: Point2,
    area_max: Point2,
    density: float,
    size_range: Tuple[float, float],
    height_profile: Optional[List[HeightTier]] = None,
    size_skew: float = 2.0,
    min_gap: float = 0.0,
    max_tries: int = 20000,
    seed: Optional[int] = None,
) -> List[Building]:
    """按覆盖率密度生成城市建筑（更接近典型城市分布）。"""
    rng = random.Random(seed)
    size_min, size_max = size_range

    if density <= 0.0 or density >= 1.0:
        raise ValueError("density must be in (0, 1).")
    if size_min <= 0 or size_max <= 0:
        raise ValueError("size_range must be positive.")

    # 默认高度分布：矮楼为主，中高楼次之，高楼最少。
    profile = height_profile or [
        (0.6, (8.0, 20.0)),
        (0.3, (20.0, 50.0)),
        (0.1, (50.0, 120.0)),
    ]

    return _generate_buildings_density(
        area_min=area_min,
        area_max=area_max,
        density=density,
        size_range=size_range,
        height_profile=profile,
        size_skew=size_skew,
        min_gap=min_gap,
        max_tries=max_tries,
        rng=rng,
    )


def generate_block_city(
    area_min: Point2,
    area_max: Point2,
    block_size: Tuple[float, float],
    road_width: float,
    block_density: float,
    size_range: Tuple[float, float],
    height_profile: Optional[List[HeightTier]] = None,
    size_skew: float = 2.0,
    min_gap: float = 0.0,
    max_tries_per_block: int = 5000,
    seed: Optional[int] = None,
) -> List[Building]:
    """生成“街区 + 道路”的结构化城市布局。

    注意：block_density 只作用在街区内部，整体覆盖率会因道路而降低。
    """
    rng = random.Random(seed)
    size_min, size_max = size_range
    if size_min <= 0 or size_max <= 0:
        raise ValueError("size_range must be positive.")
    if block_density <= 0.0 or block_density >= 1.0:
        raise ValueError("block_density must be in (0, 1).")
    if block_size[0] <= 0 or block_size[1] <= 0:
        raise ValueError("block_size must be positive.")
    if road_width < 0:
        raise ValueError("road_width must be non-negative.")

    profile = height_profile or [
        (0.6, (8.0, 20.0)),
        (0.3, (20.0, 50.0)),
        (0.1, (50.0, 120.0)),
    ]

    x_blocks = _make_block_ranges(area_min[0], area_max[0], block_size[0], road_width)
    y_blocks = _make_block_ranges(area_min[1], area_max[1], block_size[1], road_width)

    buildings: List[Building] = []
    for x0, x1 in x_blocks:
        for y0, y1 in y_blocks:
            block_min = (x0, y0)
            block_max = (x1, y1)
            block_buildings = _generate_buildings_density(
                area_min=block_min,
                area_max=block_max,
                density=block_density,
                size_range=size_range,
                height_profile=profile,
                size_skew=size_skew,
                min_gap=min_gap,
                max_tries=max_tries_per_block,
                rng=rng, 
            )
            buildings.extend(block_buildings)

    return buildings


def footprint_coverage_ratio(
    area_min: Point2,
    area_max: Point2,
    buildings: List[Building],
) -> float:
    """计算建筑占地面积 / 总面积，返回覆盖率。"""
    area_total = (area_max[0] - area_min[0]) * (area_max[1] - area_min[1])
    if area_total <= 0:
        return 0.0
    covered = sum(b.size_xy[0] * b.size_xy[1] for b in buildings)
    return covered / area_total
