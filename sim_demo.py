"""完整仿真示例：城市环境 + 采样 + 统计 + 可视化。"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

from city_env import Point2, Point3, footprint_coverage_ratio, generate_city_buildings
from los import is_los
from path_loss import path_loss_db, rx_power_dbm
from uav_trajectory import generate_uav_positions_line

# 仰角区间：(最小角度, 最大角度)
AngleBin = Tuple[float, float]


@dataclass(frozen=True)
class LinkSample:
    """存放一条链路的计算结果。"""
    uav: Point3
    user: Point3
    distance_m: float
    horizontal_m: float
    elevation_deg: float
    los: bool
    path_loss_db: float
    rx_dbm: float


def _within_area(point: Point3, area_min: Point2, area_max: Point2) -> bool:
    """判断点是否落在矩形区域内（只看 x,y)。"""
    return area_min[0] <= point[0] <= area_max[0] and area_min[1] <= point[1] <= area_max[1]


def generate_ground_users_random(
    area_min: Point2,
    area_max: Point2,
    count: int,
    seed: Optional[int] = None,
) -> List[Point3]:
    """随机生成若干地面用户位置(z=0)。"""
    rng = random.Random(seed)
    users: List[Point3] = []
    for _ in range(count):
        # 在区域内均匀撒点。
        users.append((rng.uniform(area_min[0], area_max[0]), rng.uniform(area_min[1], area_max[1]), 0.0))
    return users


def generate_pairs_angle_bins(
    area_min: Point2,
    area_max: Point2,
    altitude_range: Tuple[float, float],
    horizontal_range: Tuple[float, float],
    angle_bins: List[AngleBin],
    per_bin: int,
    seed: Optional[int] = None,
    max_tries: int = 200,
) -> List[Tuple[Point3, Point3]]:
    """按仰角分箱采样，保证低仰角到高仰角都有样本。"""
    rng = random.Random(seed)
    pairs: List[Tuple[Point3, Point3]] = []
    alt_min, alt_max = altitude_range
    d_min, d_max = horizontal_range

    for ang_min, ang_max in angle_bins:
        for _ in range(per_bin):
            for _ in range(max_tries):
                angle_deg = rng.uniform(ang_min, ang_max)
                # 仰角 = arctan(高度 / 水平距离)。
                tan_val = math.tan(math.radians(angle_deg))
                if tan_val <= 0:
                    continue
                # 由角度和水平距离区间反推高度范围。
                allowed_min = max(alt_min, d_min * tan_val)
                allowed_max = min(alt_max, d_max * tan_val)
                if allowed_min >= allowed_max:
                    continue
                altitude = rng.uniform(allowed_min, allowed_max)
                # 由仰角和高度得到水平距离。
                horizontal = altitude / tan_val

                # 随机生成一个地面用户，并在随机方向放置 UAV。
                user = (
                    rng.uniform(area_min[0], area_max[0]),
                    rng.uniform(area_min[1], area_max[1]),
                    0.0,
                )
                bearing = rng.uniform(0.0, math.tau)
                uav = (
                    user[0] + horizontal * math.cos(bearing),
                    user[1] + horizontal * math.sin(bearing),
                    altitude,
                )
                if _within_area(uav, area_min, area_max):
                    pairs.append((uav, user))
                    break

    return pairs


def generate_random_pairs(
    area_min: Point2,
    area_max: Point2,
    altitude_range: Tuple[float, float],
    horizontal_range: Tuple[float, float],
    count: int,
    seed: Optional[int] = None,
    max_tries: int = 2000,
) -> List[Tuple[Point3, Point3]]:
    """纯随机采样 UAV-用户对，并控制水平距离范围。"""
    rng = random.Random(seed)
    pairs: List[Tuple[Point3, Point3]] = []
    tries = 0
    d_min, d_max = horizontal_range
    z_min, z_max = altitude_range

    while len(pairs) < count and tries < max_tries:
        tries += 1
        user = (rng.uniform(area_min[0], area_max[0]), rng.uniform(area_min[1], area_max[1]), 0.0)
        uav = (
            rng.uniform(area_min[0], area_max[0]),
            rng.uniform(area_min[1], area_max[1]),
            rng.uniform(z_min, z_max),
        )
        horizontal = math.hypot(uav[0] - user[0], uav[1] - user[1])
        if d_min <= horizontal <= d_max:
            pairs.append((uav, user))

    return pairs


def build_link_samples(
    pairs: List[Tuple[Point3, Point3]],
    buildings,
    freq_hz: float,
    tx_power_dbm: float,
    nlos_extra_db: float,
    shadow_std_db: float,
    seed: Optional[int] = None,
) -> List[LinkSample]:
    """把 (UAV, 用户) 对转换为带统计字段的样本。"""
    rng = random.Random(seed)
    samples: List[LinkSample] = []
    for uav, user in pairs:
        # LoS 判断：只要被任一建筑遮挡就认为 NLoS。
        los_flag = is_los(uav, user, buildings)
        dx = uav[0] - user[0]
        dy = uav[1] - user[1]
        # 水平距离（x-y 平面）。
        horizontal = math.hypot(dx, dy)
        # 三维直线距离。
        distance = math.dist(uav, user)
        # 仰角 = atan(高度差 / 水平距离)。
        elevation_deg = math.degrees(math.atan2(uav[2] - user[2], horizontal))
        pl_db = path_loss_db(
            distance,
            freq_hz,
            los_flag,
            nlos_extra_db=nlos_extra_db,
            shadow_std_db=shadow_std_db,
            rng=rng,
        )
        rx_dbm = rx_power_dbm(tx_power_dbm, pl_db)
        samples.append(
            LinkSample(
                uav=uav,
                user=user,
                distance_m=distance,
                horizontal_m=horizontal,
                elevation_deg=elevation_deg,
                los=los_flag,
                path_loss_db=pl_db,
                rx_dbm=rx_dbm,
            )
        )
    return samples


def _percentile(sorted_values: List[float], q: float) -> float:
    """简单分位数计算，q 取值 [0,1]。"""
    if not sorted_values:
        return float("nan")
    idx = (len(sorted_values) - 1) * q
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return sorted_values[int(idx)]
    frac = idx - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


def summarize_samples(samples: List[LinkSample]) -> None:
    """打印样本的基本统计信息（LoS 比例、仰角范围、路径损耗范围）。"""
    if not samples:
        print("no samples.")
        return
    los_samples = [s for s in samples if s.los]
    nlos_samples = [s for s in samples if not s.los]
    los_ratio = len(los_samples) / len(samples)

    pl_all = sorted(s.path_loss_db for s in samples)
    elev_all = sorted(s.elevation_deg for s in samples)

    print(f"samples: {len(samples)}  los_ratio: {los_ratio:.2f}")
    print(f"elevation_deg: min={min(elev_all):.1f} max={max(elev_all):.1f} p10={_percentile(elev_all, 0.1):.1f} p90={_percentile(elev_all, 0.9):.1f}")
    print(f"path_loss_db: min={min(pl_all):.2f} max={max(pl_all):.2f} median={_percentile(pl_all, 0.5):.2f}")
    if los_samples:
        pl_los = sorted(s.path_loss_db for s in los_samples)
        print(f"los_path_loss_db: min={min(pl_los):.2f} max={max(pl_los):.2f} median={_percentile(pl_los, 0.5):.2f}")
    if nlos_samples:
        pl_nlos = sorted(s.path_loss_db for s in nlos_samples)
        print(f"nlos_path_loss_db: min={min(pl_nlos):.2f} max={max(pl_nlos):.2f} median={_percentile(pl_nlos, 0.5):.2f}")

    # 根据 LoS 比例给出一个非常粗略的参数建议。
    if los_ratio < 0.3:
        print("suggest: decrease density/height or reduce nlos_extra_db.")
    elif los_ratio > 0.8:
        print("suggest: increase density/height or increase nlos_extra_db.")


def write_samples_csv(path: str, samples: List[LinkSample]) -> None:
    """把样本写到 CSV 文件，便于后续分析或训练。"""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # 表头，方便 Excel/脚本读取。
        writer.writerow(
            [
                "uav_x",
                "uav_y",
                "uav_z",
                "user_x",
                "user_y",
                "distance_m",
                "horizontal_m",
                "elevation_deg",
                "los",
                "path_loss_db",
                "rx_dbm",
            ]
        )
        for s in samples:
            writer.writerow(
                [
                    f"{s.uav[0]:.3f}",
                    f"{s.uav[1]:.3f}",
                    f"{s.uav[2]:.3f}",
                    f"{s.user[0]:.3f}",
                    f"{s.user[1]:.3f}",
                    f"{s.distance_m:.3f}",
                    f"{s.horizontal_m:.3f}",
                    f"{s.elevation_deg:.3f}",
                    int(s.los),
                    f"{s.path_loss_db:.3f}",
                    f"{s.rx_dbm:.3f}",
                ]
            )


def plot_rx_vs_distance(samples: List[LinkSample], output_path: str) -> bool:
    """绘制接收功率随距离变化的散点图(LoS 蓝 / NLoS 红）。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        # 环境未安装 matplotlib 就跳过绘图。
        return False

    distances = [s.distance_m for s in samples]
    rx = [s.rx_dbm for s in samples]
    colors = ["tab:blue" if s.los else "tab:red" for s in samples]

    plt.figure(figsize=(7.0, 4.0))
    plt.scatter(distances, rx, c=colors, s=18, alpha=0.8)
    plt.xlabel("Distance (m)")
    plt.ylabel("Rx power (dBm)")
    plt.title("Rx power vs distance (LoS blue / NLoS red)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def build_trajectory_samples(
    area_min: Point2,
    area_max: Point2,
    altitude: float,
    num_points: int,
    user: Optional[Point3],
    buildings,
    freq_hz: float,
    tx_power_dbm: float,
    nlos_extra_db: float,
    shadow_std_db: float,
    seed: Optional[int] = None,
) -> List[LinkSample]:
    """沿一条直线轨迹采样无人机位置，计算与单个用户的链路。"""
    if user is None:
        # 默认把用户放在区域中心。
        user = ((area_min[0] + area_max[0]) * 0.5, (area_min[1] + area_max[1]) * 0.5, 0.0)
    uavs = generate_uav_positions_line(
        start=(area_min[0], area_min[1], altitude),
        end=(area_max[0], area_max[1], altitude),
        num_points=num_points,
    )
    pairs = [(uav, user) for uav in uavs]
    return build_link_samples(
        pairs,
        buildings,
        freq_hz=freq_hz,
        tx_power_dbm=tx_power_dbm,
        nlos_extra_db=nlos_extra_db,
        shadow_std_db=shadow_std_db,
        seed=seed,
    )


def main() -> None:
    # ===== 可调参数区 =====
    seed = 11
    area_min = (0.0, 0.0)
    area_max = (500.0, 500.0)

    # 建筑密度与高度分布（可调到更像“城市”）。
    density = 0.28
    size_range = (10.0, 40.0)
    height_profile = [
        (0.6, (8.0, 22.0)),
        (0.3, (22.0, 55.0)),
        (0.1, (55.0, 120.0)),
    ]

    # 采样范围设置（高度、水平距离、仰角分布）。
    altitude_range = (40.0, 140.0)
    horizontal_range = (30.0, 350.0)
    angle_bins = [(5.0, 15.0), (15.0, 25.0), (25.0, 35.0), (35.0, 45.0), (45.0, 60.0), (60.0, 75.0)]
    pairs_per_bin = 40
    random_pairs = 60

    # 信号相关参数（频率、发射功率、遮挡损耗）。
    freq_hz = 2.4e9
    tx_power_dbm = 20.0
    nlos_extra_db = 20.0
    shadow_std_db = 6.0

    # 生成城市建筑。
    buildings = generate_city_buildings(
        area_min=area_min,
        area_max=area_max,
        density=density,
        size_range=size_range,
        height_profile=height_profile,
        size_skew=2.2,
        min_gap=4.0,
        seed=seed,
      )
    coverage = footprint_coverage_ratio(area_min, area_max, buildings)
    print(f"buildings: {len(buildings)}  coverage: {coverage:.2f}")

    # 生成“仰角分箱 + 随机”的 UAV-用户对。
    pairs = generate_pairs_angle_bins(
        area_min=area_min,
        area_max=area_max,
        altitude_range=altitude_range,
        horizontal_range=horizontal_range,
        angle_bins=angle_bins,
        per_bin=pairs_per_bin,
        seed=seed + 1,
    )
    pairs += generate_random_pairs(
        area_min=area_min,
        area_max=area_max,
        altitude_range=altitude_range,
        horizontal_range=horizontal_range,
        count=random_pairs,
        seed=seed + 2,
    )

    # 计算所有样本的 LoS/NLoS 与路径损耗。
    samples = build_link_samples(
        pairs,
        buildings,
        freq_hz=freq_hz,
        tx_power_dbm=tx_power_dbm,
        nlos_extra_db=nlos_extra_db,
        shadow_std_db=shadow_std_db,
        seed=seed + 3,
    )
    summarize_samples(samples)

    # 输出 CSV 数据集。
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, "dataset_links.csv")
    write_samples_csv(dataset_path, samples)
    print(f"dataset_csv: {dataset_path}")

    # 选择一条直线轨迹，生成绘图样本。
    trajectory_samples = build_trajectory_samples(
        area_min=area_min,
        area_max=area_max,
        altitude=100.0,
        num_points=80,
        user=None,
        buildings=buildings,
        freq_hz=freq_hz,
        tx_power_dbm=tx_power_dbm,
        nlos_extra_db=nlos_extra_db,
        shadow_std_db=shadow_std_db,
        seed=seed + 4,
    )
    traj_csv = os.path.join(output_dir, "trajectory_links.csv")
    write_samples_csv(traj_csv, trajectory_samples)
    # 绘制“接收功率-距离”散点图，区分 LoS/NLoS。
    plot_path = os.path.join(output_dir, "rx_vs_distance.png")
    plotted = plot_rx_vs_distance(trajectory_samples, plot_path)
    if plotted:
        print(f"plot_png: {plot_path}")
    else:
        print(f"plot_skipped (matplotlib not available), trajectory_csv: {traj_csv}")


if __name__ == "__main__":
    main()
