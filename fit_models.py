# -*- coding: utf-8 -*-
"""第五周拟合分析脚本（小白级注释版）。

主要流程：
1) 读取仿真数据(NPZ 或 CSV 批次）
2) 做数据清洗与分箱统计（仰角 -> LoS 概率）
3) 拟合 LoS 概率模型(Logistic 三参数）
4) 拟合 LoS/NLoS 路径损耗模型，并计算残差与阴影衰落
5) 输出表格、JSON 与图表，便于论文写作与复核
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# 光速常量（m/s），用于 3GPP 公式中的断点距离计算
SPEED_OF_LIGHT = 3.0e8

# Al-Hourani 常见环境参数（文献常用值，可按需调整）
AL_HOURANI_PRESETS = {
    "suburban": (4.88, 0.43),
    "urban": (9.61, 0.16),
    "dense_urban": (12.08, 0.11),
    "highrise": (27.23, 0.08),
}

# 经典路径损耗模型配置（可按需改为 3GPP 参数）
CLASSIC_PATH_LOSS_MODELS = [
    {
        "name": "fspl_baseline",
        "type": "fspl_offset",
        "enabled": True,
        "freq_hz": 2.4e9,
        "nlos_offset_db": 20.0,
    },
    {
        "name": "3gpp_uma",
        "type": "3gpp_uma",
        "enabled": True,
        "freq_ghz": 2.4,
        "h_ut_m": 1.5,
    },
    {
        "name": "3gpp_placeholder",
        "type": "log_distance",
        "enabled": False,
        "los": {"A_db": None, "n": None},
        "nlos": {"A_db": None, "n": None},
    },
]

# 误差阈值（用于统计“±X dB 以内”的比例）
ERROR_THRESHOLDS_DB = [3.0, 5.0, 10.0]

def find_latest_npz(output_dir: Path) -> Optional[Path]:
    """在输出目录里找最新的 .npz 文件（作为优先数据源）。"""
    candidates = sorted(output_dir.glob("*_links.npz"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_csv_batches(output_dir: Path) -> List[Path]:
    """查找 CSV 批次文件（当 NPZ 不存在时使用）。"""
    patterns = ["*_links_*.csv", "dataset_links_*.csv", "run_big_links_*.csv"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(output_dir.glob(pattern))
    return sorted(set(files))


def load_npz_dataset(path: Path) -> Tuple[List[str], List[List[float]]]:
    """读取 NPZ 数据集，返回列名与数据行。"""
    import numpy as np

    data = np.load(path, allow_pickle=True)
    columns = [str(c) for c in data["columns"]]
    rows = data["data"].tolist()
    return columns, rows


def load_csv_datasets(paths: List[Path]) -> Tuple[List[str], List[List[float]]]:
    """读取多个 CSV 批次并合并为一个数据列表。"""
    rows: List[List[float]] = []
    columns: List[str] = []
    for path in paths:
        # 逐个文件读取并追加
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if not columns:
                columns = header
            for line in reader:
                rows.append([float(x) for x in line])
    return columns, rows


def column_index(columns: List[str]) -> Dict[str, int]:
    """把列名映射为列下标，便于按名字取数据。"""
    return {name: idx for idx, name in enumerate(columns)}


def filter_valid_rows(rows: List[List[float]], idx: Dict[str, int]) -> Tuple[List[List[float]], int]:
    """过滤异常数据(NaN/Inf、距离<=0、仰角超范围等)。"""
    bad = 0
    filtered: List[List[float]] = []
    for row in rows:
        distance = row[idx["distance_m"]]
        elevation = row[idx["elevation_deg"]]
        pl = row[idx["path_loss_db"]]
        rx = row[idx["rx_dbm"]]
        if not math.isfinite(distance) or distance <= 0:
            bad += 1
            continue
        if not math.isfinite(elevation) or elevation < 0 or elevation > 90:
            bad += 1
            continue
        if not math.isfinite(pl) or not math.isfinite(rx):
            bad += 1
            continue
        filtered.append(row)
    return filtered, bad


def bin_los_probability(
    elevations: List[float],
    los_flags: List[int],
    bin_deg: float,
    min_deg: float,
    max_deg: float,
) -> Tuple[List[float], List[float], List[int], List[float]]:
    """按仰角分箱，统计每个 bin 的 LoS 概率。"""
    edges = []
    v = min_deg
    while v <= max_deg + 1e-9:
        edges.append(v)
        v += bin_deg
    centers = [(edges[i] + edges[i + 1]) * 0.5 for i in range(len(edges) - 1)]
    counts = [0] * (len(edges) - 1)
    los_counts = [0] * (len(edges) - 1)

    for elev, los in zip(elevations, los_flags):
        if elev < min_deg or elev >= max_deg:
            continue
        bin_idx = int((elev - min_deg) // bin_deg)
        if 0 <= bin_idx < len(counts):
            counts[bin_idx] += 1
            los_counts[bin_idx] += int(los)

    probs = []
    for total, los in zip(counts, los_counts):
        probs.append(los / total if total > 0 else float("nan"))
    return centers, probs, counts, edges


def logistic_prob(theta: float, a: float, b: float, c: float) -> float:
    """Logistic LoS 概率模型：p=1/(1+a*exp(-b*(theta-c)))。"""
    return 1.0 / (1.0 + a * math.exp(-b * (theta - c)))


def al_hourani_prob(theta: float, alpha: float, beta: float) -> float:
    """Al-Hourani LoS 概率模型：p=1/(1+alpha*exp(-beta*(theta-alpha)))。"""
    return 1.0 / (1.0 + alpha * math.exp(-beta * (theta - alpha)))


def fspl_db(distance_m: float, freq_hz: float) -> float:
    """自由空间路径损耗（FSPL），单位 dB。"""
    if distance_m <= 0 or freq_hz <= 0:
        return float("nan")
    wavelength = 3.0e8 / freq_hz
    return 20.0 * math.log10(4.0 * math.pi * distance_m / wavelength)


def uma_los_path_loss(d2d_m: float, d3d_m: float, fc_ghz: float, h_bs_m: float, h_ut_m: float) -> float:
    """3GPP UMa LoS 路径损耗（TR 38.901 简化版）。"""
    if d2d_m <= 0 or d3d_m <= 0 or fc_ghz <= 0:
        return float("nan")
    if h_bs_m <= 0 or h_ut_m <= 0:
        return float("nan")
    # 断点距离 d_bp
    d_bp = 4.0 * max(h_bs_m - 1.0, 0.0) * max(h_ut_m - 1.0, 0.0) * (fc_ghz * 1e9) / SPEED_OF_LIGHT
    pl1 = 28.0 + 22.0 * math.log10(d3d_m) + 20.0 * math.log10(fc_ghz)
    if d_bp <= 0 or d2d_m <= d_bp:
        return pl1
    pl2 = (
        28.0
        + 40.0 * math.log10(d3d_m)
        + 20.0 * math.log10(fc_ghz)
        - 9.0 * math.log10(d_bp * d_bp + (h_bs_m - h_ut_m) ** 2)
    )
    return pl2


def uma_nlos_path_loss(d3d_m: float, fc_ghz: float, h_ut_m: float) -> float:
    """3GPP UMa NLoS 路径损耗（TR 38.901 简化版）。"""
    if d3d_m <= 0 or fc_ghz <= 0:
        return float("nan")
    return 13.54 + 39.08 * math.log10(d3d_m) + 20.0 * math.log10(fc_ghz) - 0.6 * (h_ut_m - 1.5)


def fit_los_probability(theta: List[float], p: List[float], weights: List[int]) -> Tuple[float, float, float]:
    """拟合 Logistic 三参数模型，优先用 SciPy,缺失时用网格搜索。"""
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except Exception:
        np = None
        curve_fit = None

    if curve_fit is not None and np is not None:
        valid = [(t, pv, w) for t, pv, w in zip(theta, p, weights) if math.isfinite(pv)]
        if not valid:
            return 10.0, 0.1, 10.0
        t_vals = np.array([v[0] for v in valid], dtype=float)
        p_vals = np.array([v[1] for v in valid], dtype=float)
        sigma = np.array([1.0 / max(v[2], 1) for v in valid], dtype=float)

        def model(t, a, b, c):
            return 1.0 / (1.0 + a * np.exp(-b * (t - c)))

        # 使用加权最小二乘进行曲线拟合
        params, _ = curve_fit(
            model,
            t_vals,
            p_vals,
            p0=(10.0, 0.1, 10.0),
            sigma=sigma,
            bounds=([0.0, 0.0, 0.0], [100.0, 5.0, 90.0]),
            maxfev=20000,
        )
        return float(params[0]), float(params[1]), float(params[2])

    # 如果没有 SciPy，就用网格搜索做一个“能用”的拟合
    best_a = 10.0
    best_b = 0.1
    best_c = 10.0
    best_loss = float("inf")

    def evaluate(a_candidates, b_candidates, c_candidates):
        nonlocal best_a, best_b, best_c, best_loss
        for a in a_candidates:
            for b in b_candidates:
                for c in c_candidates:
                    loss = 0.0
                    for t, pv, w in zip(theta, p, weights):
                        if not math.isfinite(pv):
                            continue
                        pred = logistic_prob(t, a, b, c)
                        loss += (pred - pv) ** 2 * max(w, 1)
                    if loss < best_loss:
                        best_loss = loss
                        best_a = a
                        best_b = b
                        best_c = c

    # 粗搜索：大范围扫一遍
    a_candidates = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0]
    b_candidates = [0.02 * i for i in range(1, 16)]
    c_candidates = [2.0 * i for i in range(1, 31)]
    evaluate(a_candidates, b_candidates, c_candidates)

    # 细搜索：在最优附近缩小范围
    a_min = max(0.1, best_a - 10.0)
    a_max = best_a + 10.0
    b_min = max(0.001, best_b - 0.1)
    b_max = best_b + 0.1
    c_min = max(0.0, best_c - 10.0)
    c_max = min(90.0, best_c + 10.0)

    a_candidates = [a_min + i * 1.0 for i in range(int((a_max - a_min) / 1.0) + 1)]
    b_candidates = [b_min + i * 0.01 for i in range(int((b_max - b_min) / 0.01) + 1)]
    c_candidates = [c_min + i * 1.0 for i in range(int((c_max - c_min) / 1.0) + 1)]
    evaluate(a_candidates, b_candidates, c_candidates)

    return best_a, best_b, best_c


def fit_path_loss(distances: List[float], path_losses: List[float]) -> Tuple[float, float, List[float]]:
    """拟合对数距离路径损耗模型，返回 n、PL(d0) 与残差。"""
    import numpy as np

    x = np.log10(np.array(distances, dtype=float))
    y = np.array(path_losses, dtype=float)
    # y = intercept + slope * log10(d)
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = intercept + slope * x
    residuals = (y - y_fit).tolist()
    n = float(slope / 10.0)
    pl_d0 = float(intercept)
    return n, pl_d0, residuals


def mean_std(values: List[float]) -> Tuple[float, float]:
    """计算均值与标准差（总体标准差）。"""
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def rmse_mae(y_true: List[float], y_pred: List[float], weights: Optional[List[int]] = None) -> Tuple[float, float]:
    """计算 RMSE 与 MAE(可加权）。"""
    if not y_true:
        return float("nan"), float("nan")
    if weights is None:
        weights = [1] * len(y_true)
    total = sum(weights)
    if total <= 0:
        return float("nan"), float("nan")
    mse = 0.0
    mae = 0.0
    for yt, yp, w in zip(y_true, y_pred, weights):
        err = yp - yt
        mse += (err * err) * w
        mae += abs(err) * w
    mse /= total
    mae /= total
    return math.sqrt(mse), mae


def predict_classic_path_loss(
    distances: List[float],
    los_flags: List[int],
    model: Dict,
    horizontal_m: Optional[List[float]] = None,
    uav_heights: Optional[List[float]] = None,
    user_height_m: float = 1.5,
) -> List[float]:
    """根据经典模型配置预测路径损耗（按样本逐条输出）。"""
    preds: List[float] = []
    if model.get("type") == "fspl_offset":
        freq_hz = model.get("freq_hz", 2.4e9)
        offset = model.get("nlos_offset_db", 20.0)
        for d, los in zip(distances, los_flags):
            base = fspl_db(d, freq_hz)
            preds.append(base if los == 1 else base + offset)
        return preds

    if model.get("type") == "log_distance":
        los_cfg = model.get("los", {})
        nlos_cfg = model.get("nlos", {})
        for d, los in zip(distances, los_flags):
            cfg = los_cfg if los == 1 else nlos_cfg
            A = cfg.get("A_db")
            n = cfg.get("n")
            if A is None or n is None or d <= 0:
                preds.append(float("nan"))
            else:
                preds.append(A + 10.0 * n * math.log10(d))
        return preds

    if model.get("type") == "3gpp_uma":
        if horizontal_m is None or uav_heights is None:
            return [float("nan")] * len(distances)
        fc_ghz = model.get("freq_ghz", 2.4)
        h_ut = model.get("h_ut_m", user_height_m)
        for d3d, d2d, h_bs, los in zip(distances, horizontal_m, uav_heights, los_flags):
            if not math.isfinite(d3d) or not math.isfinite(d2d) or not math.isfinite(h_bs):
                preds.append(float("nan"))
                continue
            pl_los = uma_los_path_loss(d2d, d3d, fc_ghz, h_bs, h_ut)
            pl_nlos = uma_nlos_path_loss(d3d, fc_ghz, h_ut)
            if los == 1:
                preds.append(pl_los)
            else:
                preds.append(max(pl_los, pl_nlos))
        return preds

    return [float("nan")] * len(distances)


def compute_error_metrics(errors: List[float]) -> Dict[str, float]:
    """计算误差指标（均值、RMSE、MAE）。"""
    valid = [e for e in errors if math.isfinite(e)]
    if not valid:
        return {"mean": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    mean_err = sum(valid) / len(valid)
    rmse = math.sqrt(sum(e * e for e in valid) / len(valid))
    mae = sum(abs(e) for e in valid) / len(valid)
    return {"mean": mean_err, "rmse": rmse, "mae": mae}


def percent_within(errors: List[float], threshold_db: float) -> float:
    """统计误差落在 ±threshold_db 内的比例。"""
    valid = [e for e in errors if math.isfinite(e)]
    if not valid:
        return float("nan")
    count = sum(1 for e in valid if abs(e) <= threshold_db)
    return count / len(valid)


def build_cdf(values: List[float]) -> Tuple[List[float], List[float]]:
    """生成 CDF 曲线数据（x, y）。"""
    valid = sorted(v for v in values if math.isfinite(v))
    if not valid:
        return [], []
    n = len(valid)
    xs = valid
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def plot_los_vs_elevation(
    centers: List[float],
    probs: List[float],
    a: float,
    b: float,
    c: float,
    output_path: Path,
) -> bool:
    """绘制 LoS 概率散点 + 拟合曲线。"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    xs = np.array([c for c, p in zip(centers, probs) if math.isfinite(p)], dtype=float)
    ys = np.array([p for p in probs if math.isfinite(p)], dtype=float)
    x_fit = np.linspace(0, 90, 181)
    y_fit = 1.0 / (1.0 + a * np.exp(-b * (x_fit - c)))

    plt.figure(figsize=(6.8, 4.2))
    plt.scatter(xs, ys, color="tab:blue", s=28, label="empirical")
    plt.plot(x_fit, y_fit, color="tab:orange", linewidth=2.0, label="fit")
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("p(LoS)")
    plt.title("LoS probability vs elevation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def plot_path_loss_fit(
    distances: List[float],
    path_losses: List[float],
    los_flags: List[int],
    n_los: float,
    pl0_los: float,
    n_nlos: float,
    pl0_nlos: float,
    output_path: Path,
    max_points: int = 8000,
) -> bool:
    """绘制路径损耗 vs 距离（对数坐标），叠加拟合直线。"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    data = list(zip(distances, path_losses, los_flags))
    if len(data) > max_points:
        # 样本太多时随机下采样，避免图太密
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data), size=max_points, replace=False)
        data = [data[i] for i in idx]

    d = np.array([v[0] for v in data], dtype=float)
    pl = np.array([v[1] for v in data], dtype=float)
    los = np.array([v[2] for v in data], dtype=int)

    plt.figure(figsize=(6.8, 4.6))
    plt.scatter(d[los == 1], pl[los == 1], s=10, alpha=0.5, color="tab:blue", label="LoS")
    plt.scatter(d[los == 0], pl[los == 0], s=10, alpha=0.5, color="tab:red", label="NLoS")

    x_fit = np.logspace(math.log10(max(1.0, min(distances))), math.log10(max(distances)), 100)
    y_los = pl0_los + 10.0 * n_los * np.log10(x_fit)
    y_nlos = pl0_nlos + 10.0 * n_nlos * np.log10(x_fit)
    plt.plot(x_fit, y_los, color="tab:blue", linewidth=2.0)
    plt.plot(x_fit, y_nlos, color="tab:red", linewidth=2.0)

    plt.xscale("log")
    plt.xlabel("Distance (m, log scale)")
    plt.ylabel("Path loss (dB)")
    plt.title("Path loss vs distance (fit)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def plot_histogram(values: List[float], bins: int, title: str, xlabel: str, output_path: Path) -> bool:
    """绘制直方图（用于检查分布是否合理）。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(6.2, 4.0))
    plt.hist(values, bins=bins, color="tab:blue", alpha=0.8, edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def plot_path_loss_classic(
    distances: List[float],
    path_losses: List[float],
    los_flags: List[int],
    classic_models: List[Dict],
    output_path: Path,
    max_points: int = 8000,
    uav_heights: Optional[List[float]] = None,
    user_height_m: float = 1.5,
) -> bool:
    """在散点图上叠加经典模型曲线(LoS/NLoS)。"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    data = list(zip(distances, path_losses, los_flags))
    if len(data) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data), size=max_points, replace=False)
        data = [data[i] for i in idx]

    d = np.array([v[0] for v in data], dtype=float)
    pl = np.array([v[1] for v in data], dtype=float)
    los = np.array([v[2] for v in data], dtype=int)

    plt.figure(figsize=(6.8, 4.6))
    plt.scatter(d[los == 1], pl[los == 1], s=10, alpha=0.45, color="tab:blue", label="LoS samples")
    plt.scatter(d[los == 0], pl[los == 0], s=10, alpha=0.45, color="tab:red", label="NLoS samples")

    x_fit = np.logspace(math.log10(max(1.0, min(distances))), math.log10(max(distances)), 120)
    for model in classic_models:
        name = model.get("name", "classic")
        if model.get("type") == "fspl_offset":
            freq_hz = model.get("freq_hz", 2.4e9)
            offset = model.get("nlos_offset_db", 20.0)
            y_los = [fspl_db(dv, freq_hz) for dv in x_fit]
            y_nlos = [v + offset for v in y_los]
            plt.plot(x_fit, y_los, linewidth=1.8, label=f"{name}-LoS")
            plt.plot(x_fit, y_nlos, linewidth=1.8, linestyle="--", label=f"{name}-NLoS")
        elif model.get("type") == "log_distance":
            los_cfg = model.get("los", {})
            nlos_cfg = model.get("nlos", {})
            A_los = los_cfg.get("A_db")
            n_los = los_cfg.get("n")
            A_nlos = nlos_cfg.get("A_db")
            n_nlos = nlos_cfg.get("n")
            if A_los is not None and n_los is not None:
                y_los = A_los + 10.0 * n_los * np.log10(x_fit)
                plt.plot(x_fit, y_los, linewidth=1.8, label=f"{name}-LoS")
            if A_nlos is not None and n_nlos is not None:
                y_nlos = A_nlos + 10.0 * n_nlos * np.log10(x_fit)
                plt.plot(x_fit, y_nlos, linewidth=1.8, linestyle="--", label=f"{name}-NLoS")
        elif model.get("type") == "3gpp_uma":
            fc_ghz = model.get("freq_ghz", 2.4)
            h_ut = model.get("h_ut_m", user_height_m)
            if uav_heights:
                heights = sorted(h for h in uav_heights if math.isfinite(h))
                h_bs = heights[len(heights) // 2] if heights else (h_ut + 50.0)
            else:
                h_bs = model.get("h_bs_m", h_ut + 50.0)
            h_diff = max(h_bs - h_ut, 0.0)
            d3d = x_fit
            d2d = np.sqrt(np.maximum(d3d * d3d - h_diff * h_diff, 1.0))
            y_los = [uma_los_path_loss(d2d_i, d3d_i, fc_ghz, h_bs, h_ut) for d2d_i, d3d_i in zip(d2d, d3d)]
            y_nlos = [
                max(yl, uma_nlos_path_loss(d3d_i, fc_ghz, h_ut))
                for yl, d3d_i in zip(y_los, d3d)
            ]
            plt.plot(x_fit, y_los, linewidth=1.8, label=f"{name}-LoS")
            plt.plot(x_fit, y_nlos, linewidth=1.8, linestyle="--", label=f"{name}-NLoS")

    plt.xscale("log")
    plt.xlabel("Distance (m, log scale)")
    plt.ylabel("Path loss (dB)")
    plt.title("Path loss comparison (classic models)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def plot_plos_compare(
    centers: List[float],
    probs: List[float],
    fit_params: Tuple[float, float, float],
    al_params: Tuple[float, float],
    output_path: Path,
) -> bool:
    """同图对比：经验 LoS、拟合 Logistic、Al-Hourani 曲线。"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    a, b, c = fit_params
    alpha, beta = al_params
    xs = np.array([c for c, p in zip(centers, probs) if math.isfinite(p)], dtype=float)
    ys = np.array([p for p in probs if math.isfinite(p)], dtype=float)
    x_fit = np.linspace(0, 90, 181)
    y_fit = 1.0 / (1.0 + a * np.exp(-b * (x_fit - c)))
    y_al = 1.0 / (1.0 + alpha * np.exp(-beta * (x_fit - alpha)))

    plt.figure(figsize=(6.8, 4.2))
    plt.scatter(xs, ys, color="tab:blue", s=28, label="empirical")
    plt.plot(x_fit, y_fit, color="tab:orange", linewidth=2.0, label="fitted")
    plt.plot(x_fit, y_al, color="tab:green", linewidth=2.0, label="Al-Hourani")
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("p(LoS)")
    plt.title("LoS probability comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def plot_error_cdf(
    error_dict: Dict[str, Dict[str, List[float]]],
    output_path: Path,
) -> bool:
    """绘制误差 CDF 曲线（LoS/NLoS 分开）。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(6.8, 4.2))
    for model_name, groups in error_dict.items():
        for cond_name, errors in groups.items():
            xs, ys = build_cdf([abs(e) for e in errors])
            if not xs:
                continue
            label = f"{model_name}-{cond_name}"
            plt.plot(xs, ys, linewidth=1.6, label=label)

    plt.xlabel("Absolute error (dB)")
    plt.ylabel("CDF")
    plt.title("Error CDF (classic models)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def save_binned_los(
    output_path: Path,
    bin_edges: List[float],
    centers: List[float],
    counts: List[int],
    p_emp: List[float],
    p_fit: List[float],
) -> None:
    """保存分箱统计表：每个仰角 bin 的样本数、经验 LoS 与拟合值。"""
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_min_deg", "bin_max_deg", "center_deg", "count", "p_los_empirical", "p_los_fit"])
        for i in range(len(centers)):
            writer.writerow(
                [
                    f"{bin_edges[i]:.2f}",
                    f"{bin_edges[i + 1]:.2f}",
                    f"{centers[i]:.2f}",
                    counts[i],
                    f"{p_emp[i]:.6f}" if math.isfinite(p_emp[i]) else "",
                    f"{p_fit[i]:.6f}" if math.isfinite(p_fit[i]) else "",
                ]
            )


def main() -> None:
    """主流程入口：读取数据 -> 拟合 -> 输出图表与报告。"""
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "fit_results"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 优先读 NPZ（体积小、加载快），否则读 CSV 批次
    npz_path = find_latest_npz(output_dir)
    if npz_path and npz_path.exists():
        columns, rows = load_npz_dataset(npz_path)
        input_files = [str(npz_path)]
        print(f"load_npz: {npz_path}")
    else:
        csv_files = find_csv_batches(output_dir)
        if not csv_files:
            print("no dataset files found in output/")
            return
        columns, rows = load_csv_datasets(csv_files)
        input_files = [str(p) for p in csv_files]
        print(f"load_csv_batches: {len(csv_files)} files")

    idx = column_index(columns)
    rows, bad = filter_valid_rows(rows, idx)
    print(f"samples: {len(rows)}  bad_samples: {bad}")

    # 把常用字段提取成数组，后续计算会更方便
    distances = [r[idx["distance_m"]] for r in rows]
    elevations = [r[idx["elevation_deg"]] for r in rows]
    los_flags = [int(r[idx["los"]]) for r in rows]
    path_losses = [r[idx["path_loss_db"]] for r in rows]
    rx_dbm = [r[idx["rx_dbm"]] for r in rows]
    user_height_m = 1.5
    if "uav_z" in idx:
        uav_heights = [r[idx["uav_z"]] for r in rows]
    else:
        # 由仰角与距离反推 UAV 高度（近似）
        uav_heights = [
            user_height_m + d * math.sin(math.radians(e)) for d, e in zip(distances, elevations)
        ]
    if "horizontal_m" in idx:
        horizontal_m = [r[idx["horizontal_m"]] for r in rows]
    else:
        # 若没有水平距离字段，用几何关系补全
        horizontal_m = [d * math.cos(math.radians(e)) for d, e in zip(distances, elevations)]

    # 1) LoS 概率分箱 + Logistic 拟合
    bin_deg = 5.0
    centers, probs, counts, edges = bin_los_probability(elevations, los_flags, bin_deg=bin_deg, min_deg=0.0, max_deg=90.0)
    a, b, c = fit_los_probability(centers, probs, counts)
    p_fit = [logistic_prob(center, a, b, c) if math.isfinite(p_emp) else float("nan") for center, p_emp in zip(centers, probs)]
    valid_bins = [(pe, pf, w) for pe, pf, w in zip(probs, p_fit, counts) if math.isfinite(pe)]
    rmse_los, mae_los = rmse_mae(
        [v[0] for v in valid_bins],
        [v[1] for v in valid_bins],
        [v[2] for v in valid_bins],
    )
    print(f"fit_p_los: a={a:.4f} b={b:.4f} c={c:.4f} rmse={rmse_los:.4f} mae={mae_los:.4f}")

    # Al-Hourani 经典模型对比（默认 Urban 参数）
    al_env = "urban"
    alpha, beta = AL_HOURANI_PRESETS.get(al_env, AL_HOURANI_PRESETS["urban"])
    p_al = [al_hourani_prob(center, alpha, beta) if math.isfinite(p_emp) else float("nan") for center, p_emp in zip(centers, probs)]
    valid_al = [(pe, pa, w) for pe, pa, w in zip(probs, p_al, counts) if math.isfinite(pe)]
    rmse_al, mae_al = rmse_mae(
        [v[0] for v in valid_al],
        [v[1] for v in valid_al],
        [v[2] for v in valid_al],
    )

    # 2) 分离 LoS / NLoS，分别拟合路径损耗
    los_mask = [flag == 1 for flag in los_flags]
    nlos_mask = [flag == 0 for flag in los_flags]
    los_dist = [d for d, m in zip(distances, los_mask) if m]
    los_pl = [p for p, m in zip(path_losses, los_mask) if m]
    nlos_dist = [d for d, m in zip(distances, nlos_mask) if m]
    nlos_pl = [p for p, m in zip(path_losses, nlos_mask) if m]

    n_los, pl0_los, resid_los = fit_path_loss(los_dist, los_pl)
    n_nlos, pl0_nlos, resid_nlos = fit_path_loss(nlos_dist, nlos_pl)
    _, sigma_los = mean_std(resid_los)
    _, sigma_nlos = mean_std(resid_nlos)
    resid_rmse_los, resid_mae_los = rmse_mae([0.0] * len(resid_los), resid_los)
    resid_rmse_nlos, resid_mae_nlos = rmse_mae([0.0] * len(resid_nlos), resid_nlos)

    # 3) 全局统计（整体均值/方差）
    pl_mean, pl_std = mean_std(path_losses)
    rx_mean, rx_std = mean_std(rx_dbm)
    los_ratio = sum(los_flags) / len(los_flags) if los_flags else float("nan")
 
    # 经典路径损耗模型对比
    classic_perf_rows: List[List[str]] = []
    classic_error_cdf: Dict[str, Dict[str, List[float]]] = {}
    classic_models_used: List[Dict] = []
    for model in CLASSIC_PATH_LOSS_MODELS:
        if not model.get("enabled", False):
            continue
        name = model.get("name", "classic")
        preds = predict_classic_path_loss(
            distances,
            los_flags,
            model,
            horizontal_m=horizontal_m,
            uav_heights=uav_heights,
            user_height_m=user_height_m,
        )
        errors = [p - y if math.isfinite(p) else float("nan") for p, y in zip(preds, path_losses)]
        err_los = [e for e, flag in zip(errors, los_flags) if flag == 1 and math.isfinite(e)]
        err_nlos = [e for e, flag in zip(errors, los_flags) if flag == 0 and math.isfinite(e)]

        metrics_los = compute_error_metrics(err_los)
        metrics_nlos = compute_error_metrics(err_nlos)

        classic_error_cdf[name] = {"los": err_los, "nlos": err_nlos}
        classic_models_used.append(model)

        for cond_name, metrics, errs in [
            ("los", metrics_los, err_los),
            ("nlos", metrics_nlos, err_nlos),
        ]:
            row = [
                name,
                cond_name,
                f"{metrics['mean']:.6f}",
                f"{metrics['rmse']:.6f}",
                f"{metrics['mae']:.6f}",
            ]
            for th in ERROR_THRESHOLDS_DB:
                row.append(f"{percent_within(errs, th):.6f}")
            classic_perf_rows.append(row)

    # 低仰角区域（<=15°）的差异统计
    low_diffs = []
    for center, p_emp, p_f, p_a in zip(centers, probs, p_fit, p_al):
        if math.isfinite(p_emp) and center <= 15.0:
            low_diffs.append(p_a - p_f)
    low_angle_bias = sum(low_diffs) / len(low_diffs) if low_diffs else float("nan")

    # 汇总成一个字典，方便写成 JSON 做记录/复现
    results = {
        "input_files": input_files,
        "dataset": {
            "samples": len(rows),
            "bad_samples": bad,
            "distance_min": min(distances) if distances else float("nan"),
            "distance_max": max(distances) if distances else float("nan"),
            "elevation_min": min(elevations) if elevations else float("nan"),
            "elevation_max": max(elevations) if elevations else float("nan"),
        },
        "p_los_model": {
            "formula": "p=1/(1+a*exp(-b*(theta-c)))",
            "a": a,
            "b": b,
            "c": c,
            "bin_deg": bin_deg,
            "rmse": rmse_los,
            "mae": mae_los,
        },
        "al_hourani": {
            "environment": al_env,
            "alpha": alpha,
            "beta": beta,
            "rmse": rmse_al,
            "mae": mae_al,
        },
        "path_loss": {
            "los": {
                "n": n_los,
                "pl_d0_db": pl0_los,
                "sigma_db": sigma_los,
                "rmse_db": resid_rmse_los,
                "mae_db": resid_mae_los,
            },
            "nlos": {
                "n": n_nlos,
                "pl_d0_db": pl0_nlos,
                "sigma_db": sigma_nlos,
                "rmse_db": resid_rmse_nlos,
                "mae_db": resid_mae_nlos,
            },
        },
        "classic_path_loss_models": classic_models_used,
        "low_angle_bias_al_minus_fit": low_angle_bias,
        "global": {
            "los_ratio": los_ratio,
            "path_loss_mean_db": pl_mean,
            "path_loss_std_db": pl_std,
            "rx_mean_dbm": rx_mean,
            "rx_std_dbm": rx_std,
            "samples": len(rows),
        },
    }

    # 保存参数表（CSV）
    table_csv = report_dir / "fit_params.csv"
    with table_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        writer.writerow(["a", f"{a:.6f}"])
        writer.writerow(["b", f"{b:.6f}"])
        writer.writerow(["c", f"{c:.6f}"])
        writer.writerow(["p_los_rmse", f"{rmse_los:.6f}"])
        writer.writerow(["p_los_mae", f"{mae_los:.6f}"])
        writer.writerow(["n_los", f"{n_los:.6f}"])
        writer.writerow(["pl_d0_los_db", f"{pl0_los:.6f}"])
        writer.writerow(["sigma_los_db", f"{sigma_los:.6f}"])
        writer.writerow(["rmse_los_db", f"{resid_rmse_los:.6f}"])
        writer.writerow(["mae_los_db", f"{resid_mae_los:.6f}"])
        writer.writerow(["n_nlos", f"{n_nlos:.6f}"])
        writer.writerow(["pl_d0_nlos_db", f"{pl0_nlos:.6f}"])
        writer.writerow(["sigma_nlos_db", f"{sigma_nlos:.6f}"])
        writer.writerow(["rmse_nlos_db", f"{resid_rmse_nlos:.6f}"])
        writer.writerow(["mae_nlos_db", f"{resid_mae_nlos:.6f}"])
        writer.writerow(["los_ratio", f"{los_ratio:.6f}"])
        writer.writerow(["path_loss_mean_db", f"{pl_mean:.6f}"])
        writer.writerow(["path_loss_std_db", f"{pl_std:.6f}"])

    # 保存经典模型参数表（CSV）
    classic_params_csv = report_dir / "classic_params.csv"
    with classic_params_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "parameter", "value"])
        writer.writerow(["al_hourani", "environment", al_env])
        writer.writerow(["al_hourani", "alpha", f"{alpha:.6f}"])
        writer.writerow(["al_hourani", "beta", f"{beta:.6f}"])
        for model in classic_models_used:
            name = model.get("name", "classic")
            writer.writerow([name, "type", model.get("type")])
            if model.get("type") == "fspl_offset":
                writer.writerow([name, "freq_hz", f"{model.get('freq_hz', 0.0):.6e}"])
                writer.writerow([name, "nlos_offset_db", f"{model.get('nlos_offset_db', 0.0):.6f}"])
            elif model.get("type") == "log_distance":
                los_cfg = model.get("los", {})
                nlos_cfg = model.get("nlos", {})
                writer.writerow([name, "los_A_db", los_cfg.get("A_db")])
                writer.writerow([name, "los_n", los_cfg.get("n")])
                writer.writerow([name, "nlos_A_db", nlos_cfg.get("A_db")])
                writer.writerow([name, "nlos_n", nlos_cfg.get("n")])
            elif model.get("type") == "3gpp_uma":
                writer.writerow([name, "freq_ghz", model.get("freq_ghz", 0.0)])
                writer.writerow([name, "h_ut_m", model.get("h_ut_m", 1.5)])

    # 保存经典模型误差对比表（CSV）
    classic_perf_csv = report_dir / "classic_performance.csv"
    with classic_perf_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["model", "condition", "mean_err_db", "rmse_db", "mae_db"]
        header += [f"pct_within_{int(th)}db" for th in ERROR_THRESHOLDS_DB]
        writer.writerow(header)
        for row in classic_perf_rows:
            writer.writerow(row)

    # 保存 Markdown 摘要（便于阅读）
    md_path = report_dir / "fit_summary.md"
    with md_path.open("w") as f:
        f.write("# Fit Results\n\n")
        f.write("## LoS Probability Model\n")
        f.write("p(LoS)=1/(1+a*exp(-b*(theta-c)))\n\n")
        f.write(f"- a: {a:.6f}\n")
        f.write(f"- b: {b:.6f}\n")
        f.write(f"- c: {c:.6f}\n")
        f.write(f"- RMSE: {rmse_los:.6f}\n")
        f.write(f"- MAE: {mae_los:.6f}\n\n")
        f.write("## Classic LoS Probability (Al-Hourani)\n")
        f.write(f"- env: {al_env}\n")
        f.write(f"- alpha: {alpha:.6f}\n")
        f.write(f"- beta: {beta:.6f}\n")
        f.write(f"- RMSE: {rmse_al:.6f}\n")
        f.write(f"- MAE: {mae_al:.6f}\n\n")
        f.write("## Path Loss Model (PL(d)=PL(d0)+10n*log10(d/d0), d0=1m)\n")
        f.write(f"- LoS: n={n_los:.6f}, PL(d0)={pl0_los:.6f} dB, sigma={sigma_los:.6f} dB\n")
        f.write(f"  RMSE={resid_rmse_los:.6f} dB, MAE={resid_mae_los:.6f} dB\n")
        f.write(f"- NLoS: n={n_nlos:.6f}, PL(d0)={pl0_nlos:.6f} dB, sigma={sigma_nlos:.6f} dB\n\n")
        f.write(f"  RMSE={resid_rmse_nlos:.6f} dB, MAE={resid_mae_nlos:.6f} dB\n\n")
        f.write("## Global Stats\n")
        f.write(f"- samples: {len(rows)}\n")
        f.write(f"- los_ratio: {los_ratio:.6f}\n")
        f.write(f"- path_loss_mean/std: {pl_mean:.6f}/{pl_std:.6f} dB\n")
        f.write(f"- rx_mean/std: {rx_mean:.6f}/{rx_std:.6f} dBm\n")
        f.write("\n## Inputs\n")
        for path in input_files:
            f.write(f"- {path}\n")

    # 更完整的论文版摘要
    week5_path = report_dir / "week5_summary.md"
    with week5_path.open("w") as f:
        f.write("# Week5 Summary (Model Fitting)\n\n")
        f.write("## LoS Probability Model (Logistic)\n")
        f.write("p(LoS)=1/(1+a*exp(-b*(theta-c)))\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| a | {a:.6f} |\n")
        f.write(f"| b | {b:.6f} |\n")
        f.write(f"| c | {c:.6f} |\n")
        f.write(f"| RMSE | {rmse_los:.6f} |\n")
        f.write(f"| MAE | {mae_los:.6f} |\n\n")
        f.write("## Path Loss Model (Log-distance, d0=1m)\n")
        f.write("| Condition | n | PL(d0) (dB) | sigma (dB) | RMSE (dB) | MAE (dB) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        f.write(f"| LoS | {n_los:.6f} | {pl0_los:.6f} | {sigma_los:.6f} | {resid_rmse_los:.6f} | {resid_mae_los:.6f} |\n")
        f.write(f"| NLoS | {n_nlos:.6f} | {pl0_nlos:.6f} | {sigma_nlos:.6f} | {resid_rmse_nlos:.6f} | {resid_mae_nlos:.6f} |\n\n")
        f.write("## Global Stats\n")
        f.write(f"- samples: {len(rows)}\n")
        f.write(f"- los_ratio: {los_ratio:.6f}\n")
        f.write(f"- path_loss_mean/std: {pl_mean:.6f}/{pl_std:.6f} dB\n")
        f.write(f"- rx_mean/std: {rx_mean:.6f}/{rx_std:.6f} dBm\n\n")
        f.write("## Output Files\n")
        f.write("- fit_params.csv\n")
        f.write("- fit_metrics.json\n")
        f.write("- plos_binned.csv\n")
        f.write("- classic_params.csv\n")
        f.write("- classic_performance.csv\n")
        f.write("- classic_comparison_notes.md\n")
        f.write("- los_prob_vs_elevation.png\n")
        f.write("- path_loss_fit.png\n")
        f.write("- plos_compare.png\n")
        f.write("- classic_path_loss_compare.png\n")
        f.write("- classic_error_cdf.png\n")
        f.write("- hist_elevation.png\n")
        f.write("- hist_distance.png\n")
        f.write("- residual_hist_los.png\n")
        f.write("- residual_hist_nlos.png\n\n")
        f.write("## Input Files\n")
        for path in input_files:
            f.write(f"- {path}\n")

    # 经典模型对比摘要
    classic_notes = report_dir / "classic_comparison_notes.md"
    with classic_notes.open("w") as f:
        f.write("# Classic Model Comparison Notes\n\n")
        f.write("## LoS Probability\n")
        f.write(f"- Al-Hourani env: {al_env}\n")
        f.write(f"- alpha={alpha:.6f}, beta={beta:.6f}\n")
        f.write(f"- RMSE (Al-Hourani vs empirical bins): {rmse_al:.6f}\n")
        f.write(f"- MAE  (Al-Hourani vs empirical bins): {mae_al:.6f}\n")
        f.write(f"- Low-angle bias (Al-Hourani - fitted, <=15 deg): {low_angle_bias:.6f}\n\n")

        f.write("## Classic Path Loss Models\n")
        if classic_models_used:
            f.write("Models compared:\n")
            for model in classic_models_used:
                name = model.get("name")
                if model.get("type") == "3gpp_uma":
                    fc = model.get("freq_ghz", 0.0)
                    h_ut = model.get("h_ut_m", 1.5)
                    f.write(f"- {name} (freq_ghz={fc}, h_ut_m={h_ut})\n")
                else:
                    f.write(f"- {name}\n")
        else:
            f.write("No classic path loss model enabled. Update CLASSIC_PATH_LOSS_MODELS to enable.\n")


    # 保存分箱统计表
    binned_csv = report_dir / "plos_binned.csv"
    save_binned_los(binned_csv, edges, centers, counts, probs, p_fit)

    # 保存 JSON 指标（便于后续程序复用）
    json_path = report_dir / "fit_metrics.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    # 输出图表（论文插图用）
    plot_los_vs_elevation(centers, probs, a, b, c, report_dir / "los_prob_vs_elevation.png")
    plot_path_loss_fit(
        distances,
        path_losses,
        los_flags,
        n_los,
        pl0_los,
        n_nlos,
        pl0_nlos,
        report_dir / "path_loss_fit.png",
    )
    plot_plos_compare(
        centers,
        probs,
        (a, b, c),
        (alpha, beta),
        report_dir / "plos_compare.png",
    )
    if classic_models_used:
        plot_path_loss_classic(
            distances,
            path_losses,
            los_flags,
            classic_models_used,
            report_dir / "classic_path_loss_compare.png",
            uav_heights=uav_heights,
            user_height_m=user_height_m,
        )
        plot_error_cdf(classic_error_cdf, report_dir / "classic_error_cdf.png")
    plot_histogram(elevations, 40, "Elevation distribution", "elevation (deg)", report_dir / "hist_elevation.png")
    plot_histogram(distances, 50, "Distance distribution", "distance (m)", report_dir / "hist_distance.png")
    plot_histogram(resid_los, 40, "Residual (LoS)", "PL residual (dB)", report_dir / "residual_hist_los.png")
    plot_histogram(resid_nlos, 40, "Residual (NLoS)", "PL residual (dB)", report_dir / "residual_hist_nlos.png")

    # 分块一致性检查（大数据量时，看看每块统计是否稳定）
    block_size = 5000
    block_los = []
    block_pl = []
    for start in range(0, len(rows), block_size):
        block = rows[start : start + block_size]
        if not block:
            continue
        block_los.append(sum(int(r[idx["los"]]) for r in block) / len(block))
        block_pl.append(sum(r[idx["path_loss_db"]] for r in block) / len(block))
    if block_los:
        print(
            f"block_check: blocks={len(block_los)} los_ratio[min/max]={min(block_los):.3f}/{max(block_los):.3f} "
            f"path_loss_mean[min/max]={min(block_pl):.2f}/{max(block_pl):.2f}"
        )

    print(f"reports: {report_dir}")


if __name__ == "__main__":
    main()
