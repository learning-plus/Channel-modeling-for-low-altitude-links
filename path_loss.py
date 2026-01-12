"""路径损耗与接收功率的简化模型。"""

from __future__ import annotations

import math
import random
from typing import Optional


def fspl_db(distance_m: float, freq_hz: float) -> float:
    """自由空间路径损耗（FSPL），返回 dB。"""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive.")
    if freq_hz <= 0:
        raise ValueError("freq_hz must be positive.")
    # 波长 = 光速 / 频率。
    wavelength = 3.0e8 / freq_hz
    # FSPL = 20*log10(4*pi*d/λ)
    return 20.0 * math.log10(4.0 * math.pi * distance_m / wavelength)


def path_loss_db(
    distance_m: float,
    freq_hz: float,
    is_los: bool,
    nlos_extra_db: float = 20.0,
    shadow_std_db: float = 4.0,
    rng: Optional[random.Random] = None,
) -> float:
    """大尺度路径损耗：LoS 为 FSPL，NLoS 叠加遮挡和阴影衰落。"""
    base = fspl_db(distance_m, freq_hz)
    if is_los:
        return base
    shadow = 0.0
    if shadow_std_db > 0.0:
        # 阴影衰落用高斯随机数模拟。
        rng = rng or random.Random()
        shadow = rng.gauss(0.0, shadow_std_db)
    return base + nlos_extra_db + shadow


def rx_power_dbm(tx_power_dbm: float, path_loss_db_value: float) -> float:
    """接收功率 = 发射功率 - 路径损耗（单位 dBm）。"""
    return tx_power_dbm - path_loss_db_value
