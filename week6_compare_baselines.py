# -*- coding: utf-8 -*-
"""第六周：基线对比脚本（小白级注释版）。

主要功能：
1) 读取仿真数据（NPZ/CSV）。
2) 计算统一变量：d3d、d2d、仰角、LoS 标志、观测路径损耗。
3) 生成三类基线：FSPL、Al-Hourani(Urban/Dense)、3GPP UMa（可选 UMi）。
4) 输出对比图、误差统计表、可复现配置与总结报告。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None


SPEED_OF_LIGHT = 3.0e8


def load_config(path: Path) -> Dict:
    """读取基线配置（JSON）。"""
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def find_latest_npz(output_dir: Path, pattern: str) -> Optional[Path]:
    """查找最新的 NPZ 数据文件。"""
    candidates = sorted(output_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_csv_batches(output_dir: Path, patterns: List[str]) -> List[Path]:
    """查找所有 CSV 批次文件。"""
    files: List[Path] = []
    for pattern in patterns:
        files.extend(output_dir.glob(pattern))
    return sorted(set(files))


def load_npz_dataset(path: Path) -> Tuple[List[str], List[List[float]]]:
    """读取 NPZ 数据集，返回列名与数据行。"""
    if np is None:
        raise RuntimeError("numpy is required to load npz")
    data = np.load(path, allow_pickle=True)
    columns = [str(c) for c in data["columns"]]
    rows = data["data"].tolist()
    return columns, rows


def load_csv_datasets(paths: List[Path]) -> Tuple[List[str], List[List[float]]]:
    """读取多个 CSV 批次并合并。"""
    rows: List[List[float]] = []
    columns: List[str] = []
    for path in paths:
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if not columns:
                columns = header
            for line in reader:
                rows.append([float(x) for x in line])
    return columns, rows


def column_index(columns: List[str]) -> Dict[str, int]:
    """把列名映射为列下标。"""
    return {name: idx for idx, name in enumerate(columns)}


def filter_rows(
    rows: List[List[float]],
    idx: Dict[str, int],
    quality_cfg: Dict,
) -> Tuple[List[List[float]], Dict[str, int]]:
    """过滤异常数据，并记录异常数量。"""
    counts = {
        "total": len(rows),
        "bad_total": 0,
        "bad_distance": 0,
        "bad_elevation": 0,
        "bad_path_loss": 0,
        "bad_rx": 0,
        "bad_outlier": 0,
    }
    filtered: List[List[float]] = []

    min_d = quality_cfg.get("min_distance_m")
    max_d = quality_cfg.get("max_distance_m")
    min_e = quality_cfg.get("min_elevation_deg")
    max_e = quality_cfg.get("max_elevation_deg")
    min_pl = quality_cfg.get("min_path_loss_db")
    max_pl = quality_cfg.get("max_path_loss_db")
    drop_outliers = bool(quality_cfg.get("drop_outliers", False))

    for row in rows:
        distance = row[idx["distance_m"]]
        elevation = row[idx["elevation_deg"]]
        pl = row[idx["path_loss_db"]]
        rx = row[idx["rx_dbm"]]

        bad = False
        if not math.isfinite(distance) or (min_d is not None and distance < min_d):
            counts["bad_distance"] += 1
            bad = True
        if max_d is not None and math.isfinite(distance) and distance > max_d:
            counts["bad_distance"] += 1
            bad = True
        if not math.isfinite(elevation) or (min_e is not None and elevation < min_e):
            counts["bad_elevation"] += 1
            bad = True
        if max_e is not None and math.isfinite(elevation) and elevation > max_e:
            counts["bad_elevation"] += 1
            bad = True
        if not math.isfinite(pl):
            counts["bad_path_loss"] += 1
            bad = True
        if not math.isfinite(rx):
            counts["bad_rx"] += 1
            bad = True

        outlier = False
        if min_pl is not None and math.isfinite(pl) and pl < min_pl:
            outlier = True
        if max_pl is not None and math.isfinite(pl) and pl > max_pl:
            outlier = True
        if outlier:
            counts["bad_outlier"] += 1
            if drop_outliers:
                bad = True

        if bad:
            counts["bad_total"] += 1
            continue
        filtered.append(row)

    return filtered, counts


def bin_probability(
    values: List[float],
    flags: List[int],
    bin_size: float,
    min_v: float,
    max_v: float,
    min_samples: int,
) -> Tuple[List[float], List[float], List[int], List[float]]:
    """通用分箱：返回 bin 中心、经验概率、样本数、边界。"""
    edges = []
    v = min_v
    while v <= max_v + 1e-9:
        edges.append(v)
        v += bin_size
    centers = [(edges[i] + edges[i + 1]) * 0.5 for i in range(len(edges) - 1)]
    counts = [0] * (len(edges) - 1)
    pos_counts = [0] * (len(edges) - 1)

    for val, flag in zip(values, flags):
        if val < min_v or val >= max_v:
            continue
        bin_idx = int((val - min_v) // bin_size)
        if 0 <= bin_idx < len(counts):
            counts[bin_idx] += 1
            pos_counts[bin_idx] += int(flag)

    probs = []
    for total, pos in zip(counts, pos_counts):
        if total < min_samples:
            probs.append(float("nan"))
        else:
            probs.append(pos / total)
    return centers, probs, counts, edges


def logistic_prob(theta: float, a: float, b: float, c: float) -> float:
    """Logistic LoS 概率模型。"""
    return 1.0 / (1.0 + a * math.exp(-b * (theta - c)))


def al_hourani_prob(theta: float, alpha: float, beta: float) -> float:
    """Al-Hourani LoS 概率模型。"""
    return 1.0 / (1.0 + alpha * math.exp(-beta * (theta - alpha)))


def uma_los_prob(d2d_m: float) -> float:
    """3GPP UMa LoS 概率（按 d2D）。"""
    if d2d_m <= 0:
        return 1.0
    return min(18.0 / d2d_m, 1.0) * (1.0 - math.exp(-d2d_m / 63.0)) + math.exp(-d2d_m / 63.0)


def umi_sc_los_prob(d2d_m: float) -> float:
    """3GPP UMi Street Canyon LoS 概率（按 d2D）。"""
    if d2d_m <= 0:
        return 1.0
    return min(18.0 / d2d_m, 1.0) * (1.0 - math.exp(-d2d_m / 36.0)) + math.exp(-d2d_m / 36.0)


def fspl_db(distance_m: float, freq_hz: float) -> float:
    """自由空间路径损耗（FSPL）。"""
    if distance_m <= 0 or freq_hz <= 0:
        return float("nan")
    wavelength = SPEED_OF_LIGHT / freq_hz
    return 20.0 * math.log10(4.0 * math.pi * distance_m / wavelength)


def uma_los_path_loss(d2d_m: float, d3d_m: float, fc_ghz: float, h_bs_m: float, h_ut_m: float) -> float:
    """3GPP UMa LoS 路径损耗（TR 38.901 简化版）。"""
    if d2d_m <= 0 or d3d_m <= 0 or fc_ghz <= 0:
        return float("nan")
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


def fit_logistic(theta: List[float], p: List[float], weights: List[int]) -> Tuple[float, float, float]:
    """拟合 Logistic 参数，优先用 SciPy，否则网格搜索。"""
    try:
        from scipy.optimize import curve_fit
    except Exception:
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

    a_candidates = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0]
    b_candidates = [0.02 * i for i in range(1, 16)]
    c_candidates = [2.0 * i for i in range(1, 31)]
    evaluate(a_candidates, b_candidates, c_candidates)

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


def fit_path_loss(distances: List[float], path_losses: List[float]) -> Tuple[float, float]:
    """拟合对数距离模型，返回 n 与 PL(d0)。"""
    if np is None:
        raise RuntimeError("numpy is required for fit_path_loss")
    x = np.log10(np.array(distances, dtype=float))
    y = np.array(path_losses, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    n = float(slope / 10.0)
    pl_d0 = float(intercept)
    return n, pl_d0


def rmse_mae(y_true: List[float], y_pred: List[float]) -> Tuple[float, float]:
    """RMSE 与 MAE。"""
    if not y_true:
        return float("nan"), float("nan")
    errors = [yt - yp for yt, yp in zip(y_true, y_pred) if math.isfinite(yt) and math.isfinite(yp)]
    if not errors:
        return float("nan"), float("nan")
    mse = sum(e * e for e in errors) / len(errors)
    mae = sum(abs(e) for e in errors) / len(errors)
    return math.sqrt(mse), mae


def build_cdf(values: List[float]) -> Tuple[List[float], List[float]]:
    """构建 CDF 数据。"""
    valid = sorted(v for v in values if math.isfinite(v))
    if not valid:
        return [], []
    n = len(valid)
    xs = valid
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def effective_ut_height(h_ut: float, cfg_3gpp: Dict) -> float:
    """根据高度处理方式得到有效 UT 高度。"""
    h_min, h_max = cfg_3gpp.get("ut_height_range_m", [1.5, 22.5])
    mode = cfg_3gpp.get("height_handling", "clamp")
    if mode == "fixed":
        return float(cfg_3gpp.get("fixed_ut_height_m", h_max))
    if mode == "raw":
        return float(h_ut)
    return min(max(h_ut, h_min), h_max)


def plot_plos_vs_elevation(
    centers: List[float],
    p_emp: List[float],
    fit_params: Tuple[float, float, float],
    al_urban: Dict,
    al_dense: Dict,
    metrics: Dict[str, Tuple[float, float]],
    output_path: Path,
) -> bool:
    """Fig.1：经验 LoS + 拟合 + Al-Hourani 两套基线。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    a, b, c = fit_params
    x_fit = [i for i in range(0, 91)]
    y_fit = [logistic_prob(x, a, b, c) for x in x_fit]
    y_urban = [al_hourani_prob(x, al_urban["alpha"], al_urban["beta"]) for x in x_fit]
    y_dense = [al_hourani_prob(x, al_dense["alpha"], al_dense["beta"]) for x in x_fit]

    xs = [x for x, p in zip(centers, p_emp) if math.isfinite(p)]
    ys = [p for p in p_emp if math.isfinite(p)]

    plt.figure(figsize=(6.8, 4.2))
    plt.scatter(xs, ys, color="tab:blue", s=28, label="empirical")
    fit_rmse, fit_mae = metrics.get("fit_logistic", (float("nan"), float("nan")))
    u_rmse, u_mae = metrics.get("al_hourani_urban", (float("nan"), float("nan")))
    d_rmse, d_mae = metrics.get("al_hourani_dense", (float("nan"), float("nan")))

    plt.plot(
        x_fit,
        y_fit,
        color="tab:orange",
        linewidth=2.0,
        label=f"fitted (RMSE={fit_rmse:.3f}, MAE={fit_mae:.3f})",
    )
    plt.plot(
        x_fit,
        y_urban,
        color="tab:green",
        linewidth=2.0,
        label=f"Al-Hourani Urban (RMSE={u_rmse:.3f}, MAE={u_mae:.3f})",
    )
    plt.plot(
        x_fit,
        y_dense,
        color="tab:red",
        linewidth=2.0,
        label=f"Al-Hourani Dense (RMSE={d_rmse:.3f}, MAE={d_mae:.3f})",
    )
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("p(LoS)")
    plt.title("LoS probability vs elevation (with baselines)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def plot_plos_vs_d2d_3gpp(
    centers: List[float],
    p_emp: List[float],
    output_path: Path,
    show_umi: bool,
) -> bool:
    """Fig.A1：经验 LoS(d2d) 与 3GPP UMa/UMi 对比。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x_fit = [i for i in range(1, 2001)]
    y_uma = [uma_los_prob(x) for x in x_fit]

    xs = [x for x, p in zip(centers, p_emp) if math.isfinite(p)]
    ys = [p for p in p_emp if math.isfinite(p)]

    plt.figure(figsize=(6.8, 4.2))
    plt.scatter(xs, ys, color="tab:blue", s=28, label="empirical")
    plt.plot(x_fit, y_uma, color="tab:orange", linewidth=2.0, label="3GPP UMa")
    if show_umi:
        y_umi = [umi_sc_los_prob(x) for x in x_fit]
        plt.plot(x_fit, y_umi, color="tab:green", linewidth=2.0, label="3GPP UMi-SC")
    plt.xlabel("d2D (m)")
    plt.ylabel("p(LoS)")
    plt.title("LoS probability vs d2D (3GPP reference)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def plot_path_loss_with_baselines(
    distances: List[float],
    path_losses: List[float],
    los_flags: List[int],
    n_los: float,
    pl0_los: float,
    n_nlos: float,
    pl0_nlos: float,
    curves: Dict[str, List[float]],
    output_path: Path,
    max_points: int,
) -> bool:
    """Fig.2：路径损耗散点 + 多条基线曲线。"""
    if np is None:
        return False
    try:
        import matplotlib.pyplot as plt
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

    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(d[los == 1], pl[los == 1], s=10, alpha=0.5, color="tab:blue", label="LoS samples")
    plt.scatter(d[los == 0], pl[los == 0], s=10, alpha=0.5, color="tab:red", label="NLoS samples")

    x_fit = np.logspace(math.log10(max(1.0, min(distances))), math.log10(max(distances)), 140)
    y_los = pl0_los + 10.0 * n_los * np.log10(x_fit)
    y_nlos = pl0_nlos + 10.0 * n_nlos * np.log10(x_fit)
    plt.plot(x_fit, y_los, color="tab:blue", linewidth=2.0, label="Fitted LoS")
    plt.plot(x_fit, y_nlos, color="tab:red", linewidth=2.0, label="Fitted NLoS")

    for name, y in curves.items():
        plt.plot(x_fit, y, linewidth=1.8, label=name)

    plt.xscale("log")
    plt.xlabel("Distance (m, log scale)")
    plt.ylabel("Path loss (dB)")
    plt.title("Path loss vs distance (with baselines)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def plot_residual_cdf(residuals_dict: Dict[str, List[float]], output_path: Path) -> bool:
    """Fig.3：残差 CDF（使用绝对误差）。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(6.8, 4.2))
    for name, residuals in residuals_dict.items():
        xs, ys = build_cdf([abs(r) for r in residuals])
        if xs:
            plt.plot(xs, ys, linewidth=1.6, label=name)

    plt.xlabel("|PL_obs - PL_baseline| (dB)")
    plt.ylabel("CDF")
    plt.title("Residual CDF (baselines)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def main() -> None:
    """主入口：一键生成第六周输出。"""
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "baseline_config.json"
    config = load_config(config_path)

    output_dir = base_dir / config["data"].get("output_dir", "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "week6_results"
    report_dir.mkdir(parents=True, exist_ok=True)

    npz_path = find_latest_npz(output_dir, config["data"].get("npz_pattern", "*_links.npz"))
    if npz_path and npz_path.exists():
        columns, rows = load_npz_dataset(npz_path)
        input_files = [str(npz_path)]
        print(f"load_npz: {npz_path}")
    else:
        csv_files = find_csv_batches(output_dir, config["data"].get("csv_patterns", []))
        if not csv_files:
            print("no dataset files found in output/")
            return
        columns, rows = load_csv_datasets(csv_files)
        input_files = [str(p) for p in csv_files]
        print(f"load_csv_batches: {len(csv_files)} files")

    idx = column_index(columns)
    rows, qc = filter_rows(rows, idx, config.get("quality", {}))
    print(f"samples: {len(rows)}  bad_samples: {qc.get('bad_total', 0)}")

    distances = [r[idx["distance_m"]] for r in rows]
    elevations = [r[idx["elevation_deg"]] for r in rows]
    los_flags = [int(r[idx["los"]]) for r in rows]
    path_losses = [r[idx["path_loss_db"]] for r in rows]

    user_height_m = 1.5
    if "uav_z" in idx:
        uav_heights = [r[idx["uav_z"]] for r in rows]
    else:
        uav_heights = [user_height_m + d * math.sin(math.radians(e)) for d, e in zip(distances, elevations)]

    if "horizontal_m" in idx:
        d2d = [r[idx["horizontal_m"]] for r in rows]
    else:
        d2d = [d * math.cos(math.radians(e)) for d, e in zip(distances, elevations)]

    # LoS 概率分箱（仰角）
    bin_cfg = config.get("bins", {})
    centers, p_emp, counts, edges = bin_probability(
        elevations,
        los_flags,
        bin_cfg.get("elevation_bin_deg", 5.0),
        bin_cfg.get("elevation_min_deg", 0.0),
        bin_cfg.get("elevation_max_deg", 90.0),
        bin_cfg.get("min_samples_per_bin", 20),
    )

    fit_params = fit_logistic(centers, p_emp, counts)
    a, b, c = fit_params
    p_fit = [logistic_prob(x, a, b, c) if math.isfinite(p) else float("nan") for x, p in zip(centers, p_emp)]

    al_cfg = config.get("al_hourani", {})
    al_urban = al_cfg.get("urban", {})
    al_dense = al_cfg.get("dense_urban", {})
    p_urban = [al_hourani_prob(x, al_urban["alpha"], al_urban["beta"]) if math.isfinite(p) else float("nan") for x, p in zip(centers, p_emp)]
    p_dense = [al_hourani_prob(x, al_dense["alpha"], al_dense["beta"]) if math.isfinite(p) else float("nan") for x, p in zip(centers, p_emp)]

    # LoS 概率误差表
    plos_table = report_dir / "table_metrics_plos.csv"
    plos_metrics: Dict[str, Tuple[float, float]] = {}
    with plos_table.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "region", "rmse", "mae", "valid_bins"])

        def write_plos_metrics(name: str, preds: List[float]) -> Tuple[float, float]:
            valid = [(pe, pr) for pe, pr in zip(p_emp, preds) if math.isfinite(pe) and math.isfinite(pr)]
            if valid:
                rmse, mae = rmse_mae([v[0] for v in valid], [v[1] for v in valid])
            else:
                rmse, mae = float("nan"), float("nan")
            writer.writerow([name, "all", f"{rmse:.6f}", f"{mae:.6f}", len(valid)])

            for region in config.get("angle_regions", []):
                region_name = region["name"]
                v = []
                for center, pe, pr in zip(centers, p_emp, preds):
                    if not math.isfinite(pe) or not math.isfinite(pr):
                        continue
                    if center >= region["min_deg"] and center < region["max_deg"]:
                        v.append((pe, pr))
                if v:
                    rmse_r, mae_r = rmse_mae([x[0] for x in v], [x[1] for x in v])
                else:
                    rmse_r, mae_r = float("nan"), float("nan")
                writer.writerow([name, region_name, f"{rmse_r:.6f}", f"{mae_r:.6f}", len(v)])
            return rmse, mae

        plos_metrics["fit_logistic"] = write_plos_metrics("fit_logistic", p_fit)
        plos_metrics["al_hourani_urban"] = write_plos_metrics("al_hourani_urban", p_urban)
        plos_metrics["al_hourani_dense"] = write_plos_metrics("al_hourani_dense", p_dense)

    # 路损拟合（作为“你的模型”基线）
    los_dist = [d for d, f in zip(distances, los_flags) if f == 1]
    los_pl = [p for p, f in zip(path_losses, los_flags) if f == 1]
    nlos_dist = [d for d, f in zip(distances, los_flags) if f == 0]
    nlos_pl = [p for p, f in zip(path_losses, los_flags) if f == 0]
    n_los, pl0_los = fit_path_loss(los_dist, los_pl)
    n_nlos, pl0_nlos = fit_path_loss(nlos_dist, nlos_pl)

    # 基线预测（逐样本）
    fspl_freq = config["fspl"].get("freq_hz", 2.4e9)
    fspl_pl = [fspl_db(d, fspl_freq) for d in distances]

    fit_pl = [
        pl0_los + 10.0 * n_los * math.log10(d) if f == 1 else pl0_nlos + 10.0 * n_nlos * math.log10(d)
        for d, f in zip(distances, los_flags)
    ]

    def al_hourani_pl(theta: float, env: Dict) -> Tuple[float, float, float]:
        p = al_hourani_prob(theta, env["alpha"], env["beta"])
        return p, env["eta_los_db"], env["eta_nlos_db"]

    al_urban_avg = []
    al_urban_cond = []
    al_dense_avg = []
    al_dense_cond = []
    for theta, d, f in zip(elevations, distances, los_flags):
        base = fspl_db(d, fspl_freq)
        p_u, eta_l_u, eta_n_u = al_hourani_pl(theta, al_urban)
        pl_los_u = base + eta_l_u
        pl_nlos_u = base + eta_n_u
        al_urban_avg.append(p_u * pl_los_u + (1.0 - p_u) * pl_nlos_u)
        al_urban_cond.append(pl_los_u if f == 1 else pl_nlos_u)

        p_d, eta_l_d, eta_n_d = al_hourani_pl(theta, al_dense)
        pl_los_d = base + eta_l_d
        pl_nlos_d = base + eta_n_d
        al_dense_avg.append(p_d * pl_los_d + (1.0 - p_d) * pl_nlos_d)
        al_dense_cond.append(pl_los_d if f == 1 else pl_nlos_d)

    cfg_3gpp = config.get("3gpp", {})
    fc_ghz = cfg_3gpp.get("freq_ghz", 2.4)
    h_bs = cfg_3gpp.get("bs_height_m", 25.0)
    enable_umi = bool(cfg_3gpp.get("enable_umi_sc", False))

    gpp_uma_avg = []
    gpp_uma_cond = []
    gpp_umi_avg = []
    gpp_umi_cond = []
    for d3d, d2d_i, f, h_u in zip(distances, d2d, los_flags, uav_heights):
        h_ut = effective_ut_height(h_u, cfg_3gpp)
        pl_los = uma_los_path_loss(d2d_i, d3d, fc_ghz, h_bs, h_ut)
        pl_nlos = uma_nlos_path_loss(d3d, fc_ghz, h_ut)
        pl_nlos = max(pl_los, pl_nlos)
        p_los = uma_los_prob(d2d_i)
        gpp_uma_avg.append(p_los * pl_los + (1.0 - p_los) * pl_nlos)
        gpp_uma_cond.append(pl_los if f == 1 else pl_nlos)

        if enable_umi:
            p_los_umi = umi_sc_los_prob(d2d_i)
            gpp_umi_avg.append(p_los_umi * pl_los + (1.0 - p_los_umi) * pl_nlos)
            gpp_umi_cond.append(pl_los if f == 1 else pl_nlos)

    # 路损误差表
    pl_table = report_dir / "table_metrics_pathloss.csv"
    with pl_table.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mode", "rmse_all", "mae_all", "rmse_los", "mae_los", "rmse_nlos", "mae_nlos", "samples"])

        def write_pl_metrics(name: str, mode: str, pred: List[float]):
            rmse_all, mae_all = rmse_mae(path_losses, pred)
            los_obs = [p for p, f in zip(path_losses, los_flags) if f == 1]
            los_pred = [p for p, f in zip(pred, los_flags) if f == 1]
            nlos_obs = [p for p, f in zip(path_losses, los_flags) if f == 0]
            nlos_pred = [p for p, f in zip(pred, los_flags) if f == 0]
            rmse_los, mae_los = rmse_mae(los_obs, los_pred)
            rmse_nlos, mae_nlos = rmse_mae(nlos_obs, nlos_pred)
            writer.writerow(
                [
                    name,
                    mode,
                    f"{rmse_all:.6f}",
                    f"{mae_all:.6f}",
                    f"{rmse_los:.6f}",
                    f"{mae_los:.6f}",
                    f"{rmse_nlos:.6f}",
                    f"{mae_nlos:.6f}",
                    len(pred),
                ]
            )

        write_pl_metrics("fit_model", "cond", fit_pl)
        write_pl_metrics("fspl", "avg", fspl_pl)
        write_pl_metrics("al_hourani_urban", "avg", al_urban_avg)
        write_pl_metrics("al_hourani_urban", "cond", al_urban_cond)
        write_pl_metrics("al_hourani_dense", "avg", al_dense_avg)
        write_pl_metrics("al_hourani_dense", "cond", al_dense_cond)
        write_pl_metrics("3gpp_uma", "avg", gpp_uma_avg)
        write_pl_metrics("3gpp_uma", "cond", gpp_uma_cond)
        if enable_umi:
            write_pl_metrics("3gpp_umi_sc", "avg", gpp_umi_avg)
            write_pl_metrics("3gpp_umi_sc", "cond", gpp_umi_cond)

    # 生成对比图曲线
    h_ref = float(np.median(uav_heights)) if np is not None else float(uav_heights[len(uav_heights) // 2])
    h_ref = max(h_ref, user_height_m + 1.0)
    h_diff = h_ref - user_height_m

    x_fit = np.logspace(math.log10(max(1.0, min(distances))), math.log10(max(distances)), 140)
    d2d_fit = np.sqrt(np.maximum(x_fit * x_fit - h_diff * h_diff, 1.0))
    elev_fit = [math.degrees(math.atan2(h_diff, d2d_i)) for d2d_i in d2d_fit]

    fspl_curve = [fspl_db(d, fspl_freq) for d in x_fit]

    def al_avg_curve(env: Dict) -> List[float]:
        vals = []
        for d, theta in zip(x_fit, elev_fit):
            base = fspl_db(d, fspl_freq)
            p = al_hourani_prob(theta, env["alpha"], env["beta"])
            vals.append(p * (base + env["eta_los_db"]) + (1.0 - p) * (base + env["eta_nlos_db"]))
        return vals

    al_urban_curve = al_avg_curve(al_urban)
    al_dense_curve = al_avg_curve(al_dense)

    gpp_curve = []
    gpp_umi_curve = []
    h_ut_ref = effective_ut_height(h_ref, cfg_3gpp)
    for d3d_i, d2d_i in zip(x_fit, d2d_fit):
        pl_los = uma_los_path_loss(d2d_i, d3d_i, fc_ghz, h_bs, h_ut_ref)
        pl_nlos = uma_nlos_path_loss(d3d_i, fc_ghz, h_ut_ref)
        pl_nlos = max(pl_los, pl_nlos)
        p_los = uma_los_prob(d2d_i)
        gpp_curve.append(p_los * pl_los + (1.0 - p_los) * pl_nlos)
        if enable_umi:
            p_los_umi = umi_sc_los_prob(d2d_i)
            gpp_umi_curve.append(p_los_umi * pl_los + (1.0 - p_los_umi) * pl_nlos)

    curves = {
        "FSPL": fspl_curve,
        "Al-Hourani Urban (avg)": al_urban_curve,
        "Al-Hourani Dense (avg)": al_dense_curve,
        "3GPP UMa (avg)": gpp_curve,
    }
    if enable_umi:
        curves["3GPP UMi-SC (avg)"] = gpp_umi_curve

    # 绘图
    plot_plos_vs_elevation(
        centers,
        p_emp,
        fit_params,
        al_urban,
        al_dense,
        plos_metrics,
        report_dir / "plos_vs_elevation_with_baselines.png",
    )

    plot_path_loss_with_baselines(
        distances,
        path_losses,
        los_flags,
        n_los,
        pl0_los,
        n_nlos,
        pl0_nlos,
        curves,
        report_dir / "path_loss_vs_logd_with_baselines.png",
        config.get("plot", {}).get("max_scatter_points", 8000),
    )

    residuals = {
        "fit_model": [o - p for o, p in zip(path_losses, fit_pl)],
        "al_hourani_urban": [o - p for o, p in zip(path_losses, al_urban_avg)],
        "3gpp_uma": [o - p for o, p in zip(path_losses, gpp_uma_avg)],
        "fspl": [o - p for o, p in zip(path_losses, fspl_pl)],
    }
    plot_residual_cdf(residuals, report_dir / "residual_cdf_pathloss_baselines.png")

    # 可选：3GPP LoS(d2d) 对比图
    d2d_centers, d2d_emp, _, _ = bin_probability(
        d2d,
        los_flags,
        bin_cfg.get("d2d_bin_m", 50.0),
        bin_cfg.get("d2d_min_m", 0.0),
        bin_cfg.get("d2d_max_m", 2000.0),
        bin_cfg.get("min_samples_per_bin", 20),
    )
    plot_plos_vs_d2d_3gpp(
        d2d_centers,
        d2d_emp,
        report_dir / "plos_vs_d2d_3gpp_reference.png",
        enable_umi,
    )

    # 保存配置副本
    (report_dir / "baseline_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # 关键指标记录（便于复现与追踪）
    metrics = {
        "input_files": input_files,
        "quality": qc,
        "fit_logistic": {"a": a, "b": b, "c": c},
        "path_loss_fit": {
            "n_los": n_los,
            "pl0_los": pl0_los,
            "n_nlos": n_nlos,
            "pl0_nlos": pl0_nlos,
        },
        "plos_metrics": {k: list(v) for k, v in plos_metrics.items()},
    }
    (report_dir / "week6_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 简要总结
    summary_path = report_dir / "week6_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Week6 Summary (Baselines Comparison)\n\n")
        f.write("## Assumptions\n")
        f.write("- 3GPP UMa: BS=ground (h_bs=25m), UT=UAV, height_handling={}\n".format(cfg_3gpp.get("height_handling")))
        f.write("- UMa LoS probability uses d2D (Fig.A1 gives the reference curve)\n")
        f.write(f"- UMi-SC enabled: {enable_umi}\n")
        f.write("- Al-Hourani Urban/Dense used as baseline A2G models\n\n")
        f.write("- Residual CDF uses absolute error |PL_obs - PL_baseline|\n\n")

        f.write("## Data Quality\n")
        for k, v in qc.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Output Files\n")
        f.write("- plos_vs_elevation_with_baselines.png\n")
        f.write("- path_loss_vs_logd_with_baselines.png\n")
        f.write("- residual_cdf_pathloss_baselines.png\n")
        f.write("- plos_vs_d2d_3gpp_reference.png\n")
        f.write("- table_metrics_plos.csv\n")
        f.write("- table_metrics_pathloss.csv\n")
        f.write("- baseline_config.json\n")
        f.write("- week6_metrics.json\n")
        f.write("\n## Input Files\n")
        for p in input_files:
            f.write(f"- {p}\n")

    print(f"reports: {report_dir}")


if __name__ == "__main__":
    main()
