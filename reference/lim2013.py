"""Lim, Tong & Li (2013) 悬空 CNT 薄膜热声源的精确解与近似解。

出处：C.W. Lim, L.H. Tong, Y.C. Li, "Theory of suspended carbon nanotube thinfilm as a
thermal-acoustic source", Journal of Sound and Vibration 332 (2013) 6329-6342.

本模块实现论文的：

* 精确耦合热-力学解（Eqs. 3-16）：特征根 sigma1/sigma2、d1/d2（Eq. 10）、
  常数 C1/C2（Eqs. 11-12）、热导纳 M（Eq. 14）；
* 中高频近似（Eqs. 17-20）：M ~= sigma1 ~= sqrt(j Omega / alpha)；
* 近场声压（Eq. 21 复幅、Eq. 22 RMS）与远场 Rayleigh 距离换算（Eq. 23）。

独立于 ``core``/``coupling``：不 import 本仓任何其它模型，只用于对照。

约定对齐（三处，改错任一处结果会差一个因子）：

1. 时间约定 ``x(t) = Re[x_hat exp(+j Omega t)]``（论文用 j，本仓用 1j）。论文
   Eq. (1) 的功率 ``P_in(1 - cos Omega t)`` 在该约定下复幅为 ``-P_in``。
2. 论文的 ``beta0`` 是**单侧**热损系数；本仓 ``coupling/film_ode.py`` 的
   ``linear_leak_conductance_si`` 是**双侧合计**，即 ``2*beta0``。
3. 论文的驱动是 ``P_in/(2s)``（每单位总表面积，2s 为两面之和）；本仓驱动是每单位
   **投影**膜面积的 ``P_in/s``，即论文值的 2 倍。``c_s``（论文）= ``C_A_si``（本仓）。

论文 Table 1 的常数自身有两处不自洽，复现论文图时按原值使用，与本仓对照时改用
``gas_from_physical_params`` 走仓库自己的常数：

* Table 1 给 ``alpha = 2.25e-5``，而 ``kappa/(rho0*cp) = 2.2451e-5``（差 0.2 %）；
* Table 1 的 ``rho0*cp*(gamma-1)/gamma = 100028 Pa``，而 ``P0 = 101325 Pa``（差 1.3 %）。
  后者正是本仓链路相对论文 Eq. (22) 恒定高出 1.3 % 的来源。
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, replace

from reference.constants import PhysicalParams

__all__ = [
    "TABLE1_AIR",
    "TABLE1_ARGON",
    "LimGas",
    "gas_from_physical_params",
    "characteristic_roots",
    "exact_thermal_admittance",
    "approximate_thermal_admittance",
    "areal_drive_from_power",
    "film_temperature_hat",
    "conductive_heat_flux_hat",
    "near_field_pressure_hat_from_flux",
    "exact_pressure_hat",
    "near_field_pressure_rms_exact",
    "near_field_pressure_rms_approximate",
    "rayleigh_distance_m",
    "far_field_pressure_rms",
]


@dataclass(frozen=True)
class LimGas:
    """论文 Eq. (3)/(6) 所需的气体常数（SI）。"""

    rho0_kg_m3: float
    cp_J_kgK: float
    kappa_W_mK: float
    c0_m_s: float
    alpha_m2_s: float
    gamma: float
    p0_Pa: float = 101_325.0
    T0_K: float = 300.0

    def __post_init__(self) -> None:
        for name in ("rho0_kg_m3", "cp_J_kgK", "kappa_W_mK", "c0_m_s", "alpha_m2_s", "p0_Pa", "T0_K"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.gamma) or self.gamma <= 1.0:
            raise ValueError("gamma must be finite and greater than 1")

    @property
    def alpha_consistent_m2_s(self) -> float:
        """由 kappa/(rho0*cp) 导出的热扩散率（与 Table 1 打印值可能不同）。"""

        return self.kappa_W_mK / (self.rho0_kg_m3 * self.cp_J_kgK)


# 论文 Table 1 的打印值（alpha 用论文给的 2.25e-5，复现其图时保持原样）。
TABLE1_AIR = LimGas(
    rho0_kg_m3=1.16,
    cp_J_kgK=1006.0,
    kappa_W_mK=0.0262,
    c0_m_s=347.0,
    alpha_m2_s=2.25e-5,
    gamma=1.4,
)
TABLE1_ARGON = LimGas(
    rho0_kg_m3=1.6225,
    cp_J_kgK=520.34,
    kappa_W_mK=0.0178,
    c0_m_s=323.0,
    alpha_m2_s=2.06e-5,
    gamma=1.664,
)


def gas_from_physical_params(params: PhysicalParams | None = None) -> LimGas:
    """用本仓 ``reference.constants.PhysicalParams`` 构造同一套气体常数。

    这样 LBM 与本参照的对照只剩模型差异，不含常数口径差。
    """

    p = params or PhysicalParams()
    return LimGas(
        rho0_kg_m3=p.rho0,
        cp_J_kgK=p.cp,
        kappa_W_mK=p.kg,
        c0_m_s=p.c0,
        alpha_m2_s=p.alpha0,
        gamma=p.gamma,
        p0_Pa=p.p0,
        T0_K=p.T0,
    )


def _decaying_sqrt(z: complex) -> complex:
    """返回 Re >= 0 的平方根；实部为 0 时取 Im > 0（出射波分支）。"""

    root = cmath.sqrt(z)
    if root.real < 0.0:
        root = -root
    if root.real == 0.0 and root.imag < 0.0:
        root = -root
    return root


def _coefficients(f_hz: float, gas: LimGas) -> tuple[complex, complex, complex, complex, complex]:
    """返回论文定义的 (a, b, c, d)（Eq. 10 下方）。"""

    if not math.isfinite(f_hz) or f_hz <= 0.0:
        raise ValueError("frequency must be finite and positive")
    w = 2.0 * math.pi * float(f_hz)
    a = 1j * w / gas.alpha_m2_s
    b = -1j * w / gas.kappa_W_mK
    c = gas.rho0_kg_m3 * w * w / gas.T0_K
    d = -gas.rho0_kg_m3 * w * w / gas.p0_Pa
    return a, b, c, d


def characteristic_roots(
    f_hz: float, gas: LimGas = TABLE1_AIR
) -> tuple[complex, complex, complex, complex]:
    """返回精确的 ``(sigma1, sigma2, d1, d2)``（论文 Eq. 10）。

    ``sigma1`` 是热分支（|sigma1| ~ sqrt(Omega/alpha)，衰减极快），``sigma2`` 是声分支
    （``sigma2 -> j*Omega/c0``）。两者都取 Re > 0 以保证 ``exp(-sigma*x)`` 衰减/出射。
    """

    a, b, c, d = _coefficients(f_hz, gas)
    disc = cmath.sqrt((a + d) ** 2 - 4.0 * (a * d - b * c))
    sigma1 = _decaying_sqrt(0.5 * ((a + d) + disc))
    sigma2 = _decaying_sqrt(0.5 * ((a + d) - disc))
    d1 = (sigma1 * sigma1 - a) / b
    d2 = (sigma2 * sigma2 - a) / b
    return sigma1, sigma2, d1, d2


def exact_thermal_admittance(f_hz: float, gas: LimGas = TABLE1_AIR) -> complex:
    """精确热导纳 ``M``（论文 Eq. 14）：``Q0 = kappa * M * T_s_hat``。"""

    sigma1, sigma2, d1, d2 = characteristic_roots(f_hz, gas)
    return (d1 - d2) * sigma1 * sigma2 / (d1 * sigma1 - d2 * sigma2)


def approximate_thermal_admittance(f_hz: float, gas: LimGas = TABLE1_AIR) -> complex:
    """中高频近似 ``M ~= sigma1 ~= sqrt(j Omega / alpha)``（论文 Eq. 18）。

    与本仓 ``reference.thermal_admittance.thermal_wavenumber`` 同式。
    """

    a, *_ = _coefficients(f_hz, gas)
    return _decaying_sqrt(a)


def areal_drive_from_power(power_W: float, area_m2: float) -> float:
    """论文总电功率 ``P_in`` -> 本仓约定的面功率 ``P_in/s``（W/m^2，投影面积）。

    论文正文按 ``P_in/(2s)``（两面合计）书写；本仓 ``film_rhs`` 的驱动是投影面积口径，
    即这里的 2 倍。两面分摊由 ``gas_flux_factor=2`` 与 ``G = 2*beta0`` 承担。
    """

    if area_m2 <= 0.0:
        raise ValueError("area must be positive")
    return float(power_W) / float(area_m2)


def film_temperature_hat(
    f_hz: float,
    drive_areal_hat: complex,
    c_s_J_m2K: float,
    *,
    beta0_per_side_W_m2K: float,
    gas: LimGas = TABLE1_AIR,
    exact: bool = True,
) -> complex:
    """膜温复幅 ``T_s_hat``（论文 Eq. 16，换算到本仓面功率口径）。

    论文分母 ``beta0 + kappa*M + j*omega*c_s/2``；乘 2 并与本仓 ``film_rhs``
    （``C_A dT/dt = P - 2 q - G T``，``G = 2*beta0``）对齐后为::

        T_s_hat = drive_areal_hat / (2*beta0 + 2*kappa*M + 1j*Omega*c_s)

    ``drive_areal_hat`` 是本仓口径的复面功率幅值。论文 Eq. (1) 的
    ``P_in(1-cos Omega t)`` 在该约定下 AC 复幅为 ``-P_in``，调用方需自行带符号。
    """

    if c_s_J_m2K <= 0.0:
        raise ValueError("c_s must be positive")
    if beta0_per_side_W_m2K < 0.0:
        raise ValueError("beta0 must be non-negative")
    w = 2.0 * math.pi * float(f_hz)
    admittance = (
        exact_thermal_admittance(f_hz, gas)
        if exact
        else approximate_thermal_admittance(f_hz, gas)
    )
    denominator = 2.0 * beta0_per_side_W_m2K + 2.0 * gas.kappa_W_mK * admittance + 1j * w * c_s_J_m2K
    return complex(drive_areal_hat) / denominator


def conductive_heat_flux_hat(
    f_hz: float,
    T_s_hat: complex,
    *,
    gas: LimGas = TABLE1_AIR,
    exact: bool = True,
) -> complex:
    """单侧导热热流复幅 ``Q0_hat = kappa * M * T_s_hat``（论文 Eq. 13）。"""

    admittance = (
        exact_thermal_admittance(f_hz, gas)
        if exact
        else approximate_thermal_admittance(f_hz, gas)
    )
    return gas.kappa_W_mK * admittance * complex(T_s_hat)


def near_field_pressure_hat_from_flux(q_hat_one_sided: complex, gas: LimGas = TABLE1_AIR) -> complex:
    """近场声压复幅 ``p_hat = (gamma-1) * Q0_hat / c0``（论文 Eq. 19 / A.14）。

    这是紧致热层恒等式：它把"声压"问题完全转成"壁面热流"问题。本仓 LBM 的闭箱压力
    探针不能用于此处，但壁面热流 ``q_g_one_sided_si`` 可以。
    """

    return (gas.gamma - 1.0) * complex(q_hat_one_sided) / gas.c0_m_s


def exact_pressure_hat(
    f_hz: float,
    T_s_hat: complex,
    x_m: float,
    *,
    gas: LimGas = TABLE1_AIR,
) -> complex:
    """精确声压复幅 ``p_hat(x)``（论文 Eq. 9，含热分支与声分支）。"""

    if x_m < 0.0:
        raise ValueError("x must be non-negative")
    sigma1, sigma2, d1, d2 = characteristic_roots(f_hz, gas)
    denominator = d1 * sigma1 - d2 * sigma2
    C1 = -d2 * sigma2 / denominator * complex(T_s_hat)
    C2 = d1 * sigma1 / denominator * complex(T_s_hat)
    return d1 * C1 * cmath.exp(-sigma1 * x_m) + d2 * C2 * cmath.exp(-sigma2 * x_m)


def near_field_pressure_rms_exact(
    f_hz: float,
    T_s_hat: complex,
    *,
    x_m: float = 0.0,
    gas: LimGas = TABLE1_AIR,
) -> float:
    """精确近场 RMS 声压 ``|p_hat(x)|/sqrt(2)``（论文 Eq. 24 的同一量）。"""

    return abs(exact_pressure_hat(f_hz, T_s_hat, x_m, gas=gas)) / math.sqrt(2.0)


def near_field_pressure_rms_approximate(
    f_hz: float,
    power_W: float,
    area_m2: float,
    c_s_J_m2K: float,
    *,
    beta0_per_side_W_m2K: float,
    gas: LimGas = TABLE1_AIR,
) -> float:
    """近场 RMS 声压，论文 Eq. (22) 逐式实现（自变量为总电功率与膜面积）。

    ``P_rms = P_in/(2 sqrt(2) s) * (gamma-1)/|D| * sqrt(omega/alpha) * kappa/c0``，
    ``|D| = hypot(beta0 + kappa*sqrt(omega/2alpha), kappa*sqrt(omega/2alpha) + omega*c_s/2)``。

    论文 Eq. (20) 印刷为 ``kappa*sqrt(omega/2*kappa)``，量纲不合；Eq. (22) 两处均为
    ``kappa*sqrt(omega/2*alpha)``，此处按后者（与 Eq. 18 的 M 一致）。
    """

    w = 2.0 * math.pi * float(f_hz)
    if area_m2 <= 0.0:
        raise ValueError("area must be positive")
    admittance = gas.kappa_W_mK * math.sqrt(w / (2.0 * gas.alpha_m2_s))
    denominator = math.hypot(
        beta0_per_side_W_m2K + admittance,
        admittance + 0.5 * w * c_s_J_m2K,
    )
    return (
        float(power_W)
        / (2.0 * math.sqrt(2.0) * float(area_m2))
        * (gas.gamma - 1.0)
        / denominator
        * math.sqrt(w / gas.alpha_m2_s)
        * gas.kappa_W_mK
        / gas.c0_m_s
    )


def rayleigh_distance_m(f_hz: float, area_m2: float, gas: LimGas = TABLE1_AIR) -> float:
    """Rayleigh 距离 ``R0 = s / lambda``（论文 §4.1）：近场（平面波）区厚度。"""

    if area_m2 <= 0.0:
        raise ValueError("area must be positive")
    return float(area_m2) / (gas.c0_m_s / float(f_hz))


def far_field_pressure_rms(
    f_hz: float,
    power_W: float,
    area_m2: float,
    c_s_J_m2K: float,
    x_m: float,
    *,
    beta0_per_side_W_m2K: float,
    gas: LimGas = TABLE1_AIR,
) -> float:
    """远场 RMS 声压，论文 Eq. (23)：近场值乘 ``R0/x``。"""

    if x_m <= 0.0:
        raise ValueError("x must be positive")
    near = near_field_pressure_rms_approximate(
        f_hz, power_W, area_m2, c_s_J_m2K,
        beta0_per_side_W_m2K=beta0_per_side_W_m2K, gas=gas,
    )
    return near * rayleigh_distance_m(f_hz, area_m2, gas) / float(x_m)


def with_alpha_from_kappa(gas: LimGas) -> LimGas:
    """返回 ``alpha`` 改由 ``kappa/(rho0*cp)`` 导出的同一气体（消 Table 1 的 0.2 % 差异）。"""

    return replace(gas, alpha_m2_s=gas.alpha_consistent_m2_s)
