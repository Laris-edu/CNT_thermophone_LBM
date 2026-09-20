"""半空间薄热层的解析紧致源模型：壁温复幅值 → 热膨胀泵送速度。

采用 x(t)=Re[x_hat exp(+i Omega t)]。常物性扩散温度为
T_hat(y)=T_s_hat exp(-(1+i)y/delta_T)，delta_T=sqrt(2 alpha/Omega)。
由近似等压热层的膨胀积分得到

    u_src = (1+i)/2 * Omega * delta_T * T_s_hat/T0,
    p_src = rho0*c0*u_src.

该式是单侧平面出射源，双侧热损因子属于膜 ODE，不在此重复乘 2。
模型依赖常物性、半空间、近似等压薄热层和紧致源假设。这里的单元测试
检查公式、相位和缩放；不构成当前 LBM 近场到远场的全链路验证，
也不能用同一解析公式作为自身的独立声压精度证据。
"""

from __future__ import annotations

import math

import numpy as np


def thermal_boundary_layer_thickness_m(alpha_m2_s: float, omega_rad_s: float) -> float:
    """delta_T = sqrt(2 alpha / Omega)."""

    if alpha_m2_s <= 0.0 or omega_rad_s <= 0.0:
        raise ValueError("alpha and omega must be positive")
    return math.sqrt(2.0 * alpha_m2_s / omega_rad_s)


def thermal_pumping_velocity_m_s(
    T_s_hat_K: complex,
    *,
    T0_K: float,
    omega_rad_s: float,
    alpha_m2_s: float,
) -> complex:
    """Compact-source pumping u_src = (1+i)/2 * Omega delta_T * T_s_hat/T0 (m/s, complex).

    ``T_s_hat_K`` is the complex wall-temperature amplitude in Kelvin under the frozen
    x(t)=Re[x_hat exp(+i Omega t)] convention (e.g. Level C ``T_s_hat``)."""

    delta = thermal_boundary_layer_thickness_m(alpha_m2_s, omega_rad_s)
    return complex(0.5, 0.5) * omega_rad_s * delta * (complex(T_s_hat_K) / float(T0_K))


def soft_source_pressure_amplitude_pa(
    u_src_m_s: complex,
    *,
    rho0_kg_m3: float,
    c0_m_s: float,
) -> complex:
    """One-way handoff amplitude dp_hat = Z0 * u_src for the section-11 soft source (per side)."""

    return float(rho0_kg_m3) * float(c0_m_s) * complex(u_src_m_s)


def pumping_velocity_profile_m_s(
    y_m: np.ndarray,
    T_s_hat_K: complex,
    *,
    T0_K: float,
    omega_rad_s: float,
    alpha_m2_s: float,
) -> np.ndarray:
    """Half-space in-layer velocity profile u(y) = u_src (1 - exp(-(1+i) y/delta_T)).

    The on-stack extraction fit (source-extraction probe RIG 1) uses exactly this shape plus a
    constant closed-box backflow term."""

    y = np.asarray(y_m, dtype=float)
    if np.any(y < 0.0):
        raise ValueError("y must be non-negative")
    delta = thermal_boundary_layer_thickness_m(alpha_m2_s, omega_rad_s)
    u_src = thermal_pumping_velocity_m_s(
        T_s_hat_K, T0_K=T0_K, omega_rad_s=omega_rad_s, alpha_m2_s=alpha_m2_s)
    return u_src * (1.0 - np.exp(-(1.0 + 1.0j) * y / delta))


def temperature_profile_K(
    y_m: np.ndarray,
    T_s_hat_K: complex,
    *,
    omega_rad_s: float,
    alpha_m2_s: float,
) -> np.ndarray:
    """Decaying thermal diffusion wave T_hat(y) = T_s_hat exp(-(1+i) y/delta_T)."""

    y = np.asarray(y_m, dtype=float)
    delta = thermal_boundary_layer_thickness_m(alpha_m2_s, omega_rad_s)
    return complex(T_s_hat_K) * np.exp(-(1.0 + 1.0j) * y / delta)
