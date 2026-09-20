"""SI <-> lattice-unit mapping.

This module is the single authority for transport and tau mapping.  Other
modules consume :class:`UnitMapping` and must not recompute ``nu_lu``,
``alpha_lu``, ``nu_b_lu``, ``theta_transport_lu``, ``tau21``, ``tau22`` or
``tau32`` directly.

只保留本项目实际运行的碰撞配置：二阶中心矩闭合、measured 偏应力、tau22 迹通道、
局部输运和从局部 tau32 推导的热流系数。迁移时删除了其它路线的机构（ghost_orthogonal_local*
迹策略族、四阶闭合、解析应变率偏应力、DC/A/H 退役旋钮、alpha_repair_scale 旁路
以及三个无调用的配置工厂）。旧配置若仍写着这些旋钮，``create_unit_mapping`` 按
``_RETIRED_COLLISION_VALUES`` 明确报错，不会静默按新内核重跑。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from core.lattice import default_q, default_theta_q, normalize_velocity_set


LOCAL_TAU_POLICY = "local_tau"
BULK_VISCOSITY_POLICY_DIAGNOSTIC_ZERO = "diagnostic_zero"
BULK_VISCOSITY_POLICY_SPECIFIED = "specified"
HEAT_FLUX_TAU32_RELATION = (
    "alpha_lu = theta_transport_lu * (tau32 - 0.5); "
    "local heat-flux retention = 1 - 1/tau32; "
    "local conductive readout = 1 - 1/(2*tau32); not independent transport knobs"
)
# 已退役内核的碰撞旋钮：只有当取值与本内核唯一支持的行为不同时才报错，
# 即"旧配置被静默按新内核重跑"的正是这种情况。
_RETIRED_COLLISION_VALUES: dict[str, tuple[Any, ...]] = {
    "central_moment_closure": ("second_order",),
    "high_order_relaxation": (1.0,),
    "trace_bulk_policy": ("tau22",),
    "deviatoric_stress_policy": ("measured",),
    "regularized_shear_xy_factor": (1.0,),
    "regularized_shear_normal_factor": (1.0,),
    "acoustic_simplified_collision": (False,),
    "dispersion_correction_enabled": (False,),
    "acoustic_phase_correction_enabled": (False,),
    "seam_aware_bottom_rows": (0,),
    "seam_aware_top_rows": (0,),
    "seam_aware_taper_rows": (0,),
}
# 不再接受固定热流系数，也不静默忽略旧标定值。参考态派生值仍写入元数据。
_RETIRED_HEAT_FLUX_KEYS = (
    "regularized_heat_flux_factor", "regularized_heat_flux_factor_policy",
    "conductive_heat_flux_moment_factor", "conductive_heat_flux_moment_factor_policy",
    "conductive_heat_flux_galilean_correction_factor", "heat_flux_retention_policy",
)


@dataclass(frozen=True)
class PhysicalScales:
    T0_K: float = 300.0
    p0_Pa: float = 101_325.0
    rho0_kg_m3: float = 1.177
    c0_m_s: float = 347.0
    gamma: float = 1.4
    cp_J_kgK: float = 1005.0
    kg_W_mK: float = 0.0263
    nu0_m2_s: float = 1.57e-5
    alpha0_m2_s: float = 2.2233775895e-5
    Pr: float = 0.7061328707


@dataclass(frozen=True)
class LatticeScales:
    velocity_set: str = "D2Q37"
    Q: int = 37
    dx_m: float = 4.0e-6
    dt_s: float = 3.0e-9
    theta_q_lu: float = 0.6979533220196852
    theta_ref_policy: str = "physical_sound_speed"
    theta_transport_policy: str = "theta_ref_lu"
    rho_ref_lu: float = 1.0
    D: int = 2
    S: int = 3


@dataclass(frozen=True)
class CollisionScales:
    bulk_viscosity_policy: str = BULK_VISCOSITY_POLICY_DIAGNOSTIC_ZERO
    nu_b_lu: float | str = "auto"
    trace_bulk_scale: float = 1.0
    # 以下热流系数仅记录参考态，不是运行时可调参数。
    regularized_heat_flux_factor_policy: str = LOCAL_TAU_POLICY
    regularized_heat_flux_factor: float = 0.0
    regularized_heat_flux_f_fraction: float = 0.5714285714285714
    conductive_heat_flux_moment_factor_policy: str = LOCAL_TAU_POLICY
    conductive_heat_flux_moment_factor: float = 1.0
    conductive_heat_flux_galilean_correction_factor: float = 0.0


@dataclass(frozen=True)
class TransportScales:
    """局部输运本构；单位换算始终固定，cp 由固定 D/S 和温度单位约束。"""

    mode: str = "local"
    model: str = "sutherland"
    sutherland_mu_K: float = 111.0
    sutherland_k_K: float = 194.0


@dataclass(frozen=True)
class LocalTransport:
    """碰撞前宏观状态对应的逐格点系数；不写回参考 UnitMapping。"""

    nu_lu: np.ndarray
    alpha_lu: np.ndarray
    tau21: np.ndarray
    tau22: np.ndarray
    tau32: np.ndarray
    heat_flux_retention: np.ndarray
    heat_flux_readout: np.ndarray


@dataclass(frozen=True)
class UnitMapping:
    physical: PhysicalScales
    lattice: LatticeScales
    collision: CollisionScales
    c0_lu: float
    theta_ref_lu: float
    theta_transport_lu: float
    nu_lu: float
    alpha_lu: float
    Pr_lu: float
    nu_b_lu: float
    tau21: float
    tau22: float
    tau32: float
    rho_scale: float
    velocity_scale: float
    pressure_scale: float
    temperature_scale: float
    heat_flux_scale: float
    heat_flux_scale_definition: str
    transport: TransportScales = TransportScales()

    @property
    def gamma_from_degrees(self) -> float:
        return 1.0 + 2.0 / (self.lattice.D + self.lattice.S)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "transport_mode": self.transport.mode,
            "transport_model": self.transport.model,
            "transport_sutherland_mu_K": self.transport.sutherland_mu_K,
            "transport_sutherland_k_K": self.transport.sutherland_k_K,
            "transport_bulk_law": "constant_bulk_to_shear_ratio",
            "tau_metadata_scope": "reference_state",
            "velocity_set": self.lattice.velocity_set,
            "Q": self.lattice.Q,
            "D": self.lattice.D,
            "S": self.lattice.S,
            "array_layout": "c=(Q,D) columns=(cx,cy); w=(Q,); f,g=(...,Q); u,q=(...,D)",
            "theta_q_lu": self.lattice.theta_q_lu,
            "theta_ref_lu": self.theta_ref_lu,
            "theta_transport_lu": self.theta_transport_lu,
            "dx_m": self.lattice.dx_m,
            "dt_s": self.lattice.dt_s,
            "T0_K": self.physical.T0_K,
            "p0_Pa": self.physical.p0_Pa,
            "rho0_kg_m3": self.physical.rho0_kg_m3,
            "c0_m_s": self.physical.c0_m_s,
            "gamma_target": self.physical.gamma,
            "cp_J_kgK": self.physical.cp_J_kgK,
            "kg_W_mK": self.physical.kg_W_mK,
            "nu0_m2_s": self.physical.nu0_m2_s,
            "alpha0_m2_s": self.physical.alpha0_m2_s,
            "Pr_target": self.physical.Pr,
            "bulk_viscosity_policy": self.collision.bulk_viscosity_policy,
            # 迹通道在迁移后只剩 tau22 一种；显式记录以便与旧元数据区分。
            "trace_bulk_policy": "tau22",
            "trace_bulk_scale": self.collision.trace_bulk_scale,
            "nu_lu": self.nu_lu,
            "alpha_lu": self.alpha_lu,
            "nu_b_lu": self.nu_b_lu,
            "Pr_lu": self.Pr_lu,
            "tau21": self.tau21,
            "tau22": self.tau22,
            "tau32": self.tau32,
            "heat_flux_tau32_relation": HEAT_FLUX_TAU32_RELATION,
            "regularized_heat_flux_factor_policy": self.collision.regularized_heat_flux_factor_policy,
            "regularized_heat_flux_factor": self.collision.regularized_heat_flux_factor,
            "regularized_heat_flux_f_fraction": self.collision.regularized_heat_flux_f_fraction,
            "conductive_heat_flux_moment_factor_policy": (
                self.collision.conductive_heat_flux_moment_factor_policy
            ),
            "conductive_heat_flux_moment_factor": self.collision.conductive_heat_flux_moment_factor,
            "conductive_heat_flux_galilean_correction_factor": (
                self.collision.conductive_heat_flux_galilean_correction_factor
            ),
            "rho_scale": self.rho_scale,
            "velocity_scale": self.velocity_scale,
            "pressure_scale": self.pressure_scale,
            "temperature_scale": self.temperature_scale,
            "heat_flux_scale": self.heat_flux_scale,
            "heat_flux_scale_definition": self.heat_flux_scale_definition,
        }


def _physical_from_config(config: dict[str, Any] | None) -> PhysicalScales:
    physical = (config or {}).get("physical", {})
    alpha = physical.get("alpha0_m2_s", None)
    if alpha is None:
        alpha = physical.get("kg_W_mK", 0.0263) / (
            physical.get("rho0_kg_m3", 1.177) * physical.get("cp_J_kgK", 1005.0)
        )
    pr = physical.get("Pr", physical.get("nu0_m2_s", 1.57e-5) / alpha)
    return PhysicalScales(
        T0_K=float(physical.get("T0_K", 300.0)),
        p0_Pa=float(physical.get("p0_Pa", 101_325.0)),
        rho0_kg_m3=float(physical.get("rho0_kg_m3", 1.177)),
        c0_m_s=float(physical.get("c0_m_s", 347.0)),
        gamma=float(physical.get("gamma", 1.4)),
        cp_J_kgK=float(physical.get("cp_J_kgK", 1005.0)),
        kg_W_mK=float(physical.get("kg_W_mK", 0.0263)),
        nu0_m2_s=float(physical.get("nu0_m2_s", 1.57e-5)),
        alpha0_m2_s=float(alpha),
        Pr=float(pr),
    )


def _lattice_from_config(config: dict[str, Any] | None) -> LatticeScales:
    lattice = (config or {}).get("lattice", {})
    velocity_set = normalize_velocity_set(lattice.get("velocity_set"))
    return LatticeScales(
        velocity_set=velocity_set,
        Q=int(lattice.get("Q", default_q(velocity_set))),
        dx_m=float(lattice.get("dx_m", 4.0e-6)),
        dt_s=float(lattice.get("dt_s", 3.0e-9)),
        theta_q_lu=float(lattice.get("theta_q_lu", default_theta_q(velocity_set))),
        theta_ref_policy=str(lattice.get("theta_ref_policy", "physical_sound_speed")),
        theta_transport_policy=str(lattice.get("theta_transport_policy", "theta_ref_lu")),
        rho_ref_lu=float(lattice.get("rho_ref_lu", 1.0)),
        D=int(lattice.get("D", 2)),
        S=int(lattice.get("S", 3)),
    )


def _assert_no_retired_collision_values(collision: dict[str, Any]) -> None:
    for key, supported in _RETIRED_COLLISION_VALUES.items():
        if key not in collision:
            continue
        value = collision[key]
        if any(value == item for item in supported):
            continue
        raise ValueError(
            f"collision.{key}={value!r} 属于已退役内核的配置，本内核只支持 {supported}。"
            "旧配置不能在当前内核上静默重跑：历史结果请用旧仓对应 commit 复现，"
            "或改用当前基线 YAML。"
        )


def _collision_from_config(config: dict[str, Any] | None) -> CollisionScales:
    collision = (config or {}).get("collision", {})
    _assert_no_retired_collision_values(collision)

    for key in _RETIRED_HEAT_FLUX_KEYS:
        if key in collision:
            raise ValueError(
                f"collision.{key} is retired; heat-flux coefficients are derived from local tau32. "
                "Use the current local-transport baseline; reproduce historical runs at their original commit."
            )

    return CollisionScales(
        bulk_viscosity_policy=str(
            collision.get("bulk_viscosity_policy", BULK_VISCOSITY_POLICY_DIAGNOSTIC_ZERO)
        ),
        nu_b_lu=collision.get("nu_b_lu", "auto"),
        trace_bulk_scale=float(collision.get("trace_bulk_scale", 1.0)),
        regularized_heat_flux_f_fraction=float(
            collision.get("regularized_heat_flux_f_fraction", 0.5714285714285714)
        ),
    )


def tau32_from_alpha_lu(alpha_lu: float, theta_transport_lu: float) -> float:
    """Return tau32 from conductive thermal diffusivity in lattice units."""

    return 0.5 + float(alpha_lu) / float(theta_transport_lu)


def heat_flux_f_fraction_from_degrees(D: int, S: int) -> float:
    """Return the f-channel share of conductive enthalpy flux.

    For the current D=2, S=3 gas, this gives (D+2)/(D+S+2)=4/7; the remaining
    S/(D+S+2)=3/7 is carried by the g-channel internal-energy flux.
    """

    return (float(D) + 2.0) / (float(D) + float(S) + 2.0)


def local_transport_coefficients(
    mapping: UnitMapping, rho_lu: np.ndarray, theta_lu: np.ndarray,
) -> LocalTransport:
    """由局部 rho/T 计算本构与离散松弛时间，不修改单位或求积温度。

    mu0=rho0*nu0, k0=physical.kg_W_mK；constant 固定 mu/k 而非 nu/alpha。
    Sutherland 分别使用黏度与导热常数，cp 固定；非零体黏度保持参考
    bulk/shear 比。物理热流系数 1-1/(2*tau32) 对应碰撞前布居。
    """
    if mapping.transport.mode != "local":
        raise ValueError("local transport coefficients require transport.mode=local")
    rho, theta = np.broadcast_arrays(np.asarray(rho_lu, dtype=float), np.asarray(theta_lu, dtype=float))
    if not np.all(np.isfinite(rho) & (rho > 0)):
        raise ValueError("local transport requires finite positive density")
    if not np.all(np.isfinite(theta) & (theta > 0)):
        raise ValueError("local transport requires finite positive temperature")
    tr = theta / mapping.theta_ref_lu
    rho_ratio = rho / mapping.lattice.rho_ref_lu
    if mapping.transport.model == "constant":
        mu_ratio = k_ratio = np.ones_like(tr)
    elif mapping.transport.model == "sutherland":
        T0 = mapping.physical.T0_K
        mu_s = mapping.transport.sutherland_mu_K
        k_s = mapping.transport.sutherland_k_K
        mu_ratio = tr**1.5 * (T0 + mu_s) / (T0 * tr + mu_s)
        k_ratio = tr**1.5 * (T0 + k_s) / (T0 * tr + k_s)
    else:
        raise ValueError(f"unknown transport model: {mapping.transport.model}")
    nu = mapping.nu_lu * mu_ratio / rho_ratio
    alpha = mapping.alpha_lu * k_ratio / rho_ratio
    tau21 = 0.5 + nu / theta
    tau32 = 0.5 + alpha / theta
    if mapping.nu_b_lu == 0.0:
        tau22 = np.full_like(theta, 0.5)
    else:
        bulk = mapping.nu_b_lu * mu_ratio / rho_ratio
        D, S = mapping.lattice.D, mapping.lattice.S
        tau22 = 0.5 + bulk / (2.0 * S * theta / (D * (D + S)))
    if not np.all(np.isfinite(tau21) & (tau21 > 0.5) & np.isfinite(tau32) & (tau32 > 0.5)
                  & np.isfinite(tau22) & (tau22 >= 0.5)):
        raise ValueError("local transport produced non-finite or inadmissible relaxation times; no clipping applied")
    return LocalTransport(nu, alpha, tau21, tau22, tau32, 1.0 - 1.0 / tau32, 1.0 - 0.5 / tau32)


def _validate_local_transport_mapping(mapping: UnitMapping) -> None:
    t, p = mapping.transport, mapping.physical
    if t.mode != "local":
        raise ValueError("transport.mode must be local; frozen transport is retired")
    if t.model not in {"constant", "sutherland"}:
        raise ValueError(f"unknown transport model: {t.model}")
    if any(not np.isfinite(v) or v < 0 for v in (t.sutherland_mu_K, t.sutherland_k_K)):
        raise ValueError("transport Sutherland constants must be finite and nonnegative")
    if not np.isclose(mapping.theta_transport_lu, mapping.theta_ref_lu, rtol=1e-12, atol=0):
        raise ValueError("local transport requires theta_transport_lu=theta_ref_lu at the reference state")
    if any(not np.isfinite(v) or v <= 0 for v in (p.T0_K, p.rho0_kg_m3, p.nu0_m2_s, p.kg_W_mK, p.cp_J_kgK)):
        raise ValueError("local transport requires finite positive reference properties")
    cp_model = (mapping.lattice.D + mapping.lattice.S + 2) / 2 * mapping.velocity_scale**2 / mapping.temperature_scale
    if not np.isclose(p.cp_J_kgK, cp_model, rtol=1e-10, atol=0):
        raise ValueError(f"local transport cp must match D/S and temperature mapping: cp_J_kgK={cp_model:.15g}")
    alpha = p.kg_W_mK / (p.rho0_kg_m3 * p.cp_J_kgK) * mapping.lattice.dt_s / mapping.lattice.dx_m**2
    if not np.isclose(mapping.alpha_lu, alpha, rtol=1e-9, atol=0):
        raise ValueError("local transport alpha must equal k/(rho*cp)")
    local_transport_coefficients(mapping, mapping.lattice.rho_ref_lu, mapping.theta_ref_lu)


def _resolve_collision_closure(
    collision: CollisionScales, tau32: float,
) -> CollisionScales:
    # 这里只登记参考态；运行中全部热流系数从同一局部 tau32 产生。
    return replace(
        collision,
        regularized_heat_flux_factor=1.0 - 1.0 / tau32,
        conductive_heat_flux_moment_factor=1.0 - 0.5 / tau32,
    )


def create_unit_mapping(config: dict[str, Any] | None = None) -> UnitMapping:
    """Create the mapping from an explicit local-transport configuration.

    Missing transport.mode was the legacy frozen default. Reject it instead of
    silently reinterpreting historical configurations with different physics.
    """

    transport_config = dict((config or {}).get("transport", {}) or {})
    if transport_config.get("mode") != "local":
        raise ValueError(
            "transport.mode must explicitly be local; frozen transport and its implicit default are retired. "
            "Use configs/gas_air_10k_d2q37_qmatched_local_transport.yaml."
        )

    physical = _physical_from_config(config)
    lattice = _lattice_from_config(config)
    collision = _collision_from_config(config)
    transport = TransportScales(**transport_config)

    c0_lu = physical.c0_m_s * lattice.dt_s / lattice.dx_m
    if lattice.theta_ref_policy == "quadrature_matched":
        theta_ref_lu = lattice.theta_q_lu
    elif lattice.theta_ref_policy == "physical_sound_speed":
        theta_ref_lu = c0_lu**2 / physical.gamma
    else:
        theta_ref_lu = float((config or {}).get("lattice", {}).get("theta_ref_lu"))

    if lattice.theta_transport_policy == "theta_q_lu":
        theta_transport_lu = lattice.theta_q_lu
    elif lattice.theta_transport_policy == "theta_ref_lu":
        theta_transport_lu = theta_ref_lu
    else:
        theta_transport_lu = float((config or {}).get("lattice", {}).get("theta_transport_lu"))

    nu_lu = physical.nu0_m2_s * lattice.dt_s / lattice.dx_m**2
    alpha_lu = physical.alpha0_m2_s * lattice.dt_s / lattice.dx_m**2
    Pr_lu = nu_lu / alpha_lu

    policy = collision.bulk_viscosity_policy
    if policy == BULK_VISCOSITY_POLICY_DIAGNOSTIC_ZERO:
        nu_b_lu = 0.0
    elif policy == BULK_VISCOSITY_POLICY_SPECIFIED:
        if collision.nu_b_lu == "auto":
            nu_b_lu = 0.0
        else:
            nu_b_lu = float(collision.nu_b_lu)
    else:
        raise ValueError(f"unknown bulk_viscosity_policy: {policy}")

    tau21 = 0.5 + nu_lu / theta_transport_lu
    tau32 = tau32_from_alpha_lu(alpha_lu, theta_transport_lu)
    collision = _resolve_collision_closure(collision, tau32)
    if nu_b_lu == 0.0:
        tau22 = 0.5
    else:
        factor = 2.0 * lattice.S * theta_transport_lu / (lattice.D * (lattice.D + lattice.S))
        tau22 = 0.5 + nu_b_lu / factor

    rho_scale = physical.rho0_kg_m3 / lattice.rho_ref_lu
    velocity_scale = lattice.dx_m / lattice.dt_s
    pressure_scale = rho_scale * velocity_scale**2
    temperature_scale = physical.T0_K / theta_ref_lu
    heat_flux_scale = rho_scale * velocity_scale**3
    heat_flux_scale_definition = (
        "q_phys = q_lu * rho_scale * (dx_m/dt_s)^3; exported q_lu is conductive heat flux "
        "after the local pre-collision readout factor 1 - 1/(2*tau32)"
    )

    mapping = UnitMapping(
        physical=physical,
        lattice=lattice,
        collision=collision,
        c0_lu=c0_lu,
        theta_ref_lu=theta_ref_lu,
        theta_transport_lu=theta_transport_lu,
        nu_lu=nu_lu,
        alpha_lu=alpha_lu,
        Pr_lu=Pr_lu,
        nu_b_lu=nu_b_lu,
        tau21=tau21,
        tau22=tau22,
        tau32=tau32,
        rho_scale=rho_scale,
        velocity_scale=velocity_scale,
        pressure_scale=pressure_scale,
        temperature_scale=temperature_scale,
        heat_flux_scale=heat_flux_scale,
        heat_flux_scale_definition=heat_flux_scale_definition,
        transport=transport,
    )
    validate_unit_mapping(mapping)
    return mapping


def validate_unit_mapping(mapping: UnitMapping) -> None:
    """Validate non-negotiable mapping invariants."""

    _validate_local_transport_mapping(mapping)

    if mapping.lattice.Q != default_q(mapping.lattice.velocity_set):
        raise ValueError("Q is inconsistent with velocity_set")
    if not np.isclose(
        mapping.lattice.theta_q_lu,
        default_theta_q(mapping.lattice.velocity_set),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("theta_q_lu is inconsistent with velocity_set")
    if not np.isclose(mapping.Pr_lu, mapping.nu_lu / mapping.alpha_lu, rtol=0.0, atol=1.0e-15):
        raise ValueError("Pr_lu is inconsistent with nu_lu/alpha_lu")
    if not np.isclose(mapping.gamma_from_degrees, mapping.physical.gamma, rtol=0.0, atol=1.0e-12):
        raise ValueError("D/S degrees of freedom do not reconstruct target gamma")
    if mapping.physical.Pr < 1.0 and not mapping.tau32 > mapping.tau21:
        raise ValueError("air Pr<1 requires tau32 > tau21 for this mapping")
    if mapping.collision.bulk_viscosity_policy == BULK_VISCOSITY_POLICY_DIAGNOSTIC_ZERO and mapping.nu_b_lu != 0.0:
        raise ValueError("diagnostic_zero bulk policy requires nu_b_lu=0")
    if not np.isfinite(mapping.collision.trace_bulk_scale):
        raise ValueError("trace_bulk_scale must be finite")
    if not 0.0 <= mapping.collision.regularized_heat_flux_f_fraction <= 1.0:
        raise ValueError("regularized_heat_flux_f_fraction must be in [0, 1]")
    expected_f_fraction = heat_flux_f_fraction_from_degrees(mapping.lattice.D, mapping.lattice.S)
    if not np.isclose(
        mapping.collision.regularized_heat_flux_f_fraction,
        expected_f_fraction,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("regularized_heat_flux_f_fraction must match the D/S enthalpy split")
    if not np.isfinite(mapping.collision.regularized_heat_flux_factor):
        raise ValueError("regularized_heat_flux_factor must be finite")
    if mapping.collision.conductive_heat_flux_moment_factor <= 0.0:
        raise ValueError("conductive_heat_flux_moment_factor must be positive")
    if not np.isfinite(mapping.collision.conductive_heat_flux_galilean_correction_factor):
        raise ValueError("conductive_heat_flux_galilean_correction_factor must be finite")


def pressure_lu_to_phys(p_lu: np.ndarray, mapping: UnitMapping) -> np.ndarray:
    return np.asarray(p_lu) * mapping.pressure_scale


def temperature_lu_to_phys(theta_lu: np.ndarray, mapping: UnitMapping) -> np.ndarray:
    return np.asarray(theta_lu) * mapping.temperature_scale


def temperature_phys_to_lu(T_phys: np.ndarray, mapping: UnitMapping) -> np.ndarray:
    return np.asarray(T_phys) / mapping.temperature_scale


def heat_flux_lu_to_phys(q_lu: np.ndarray, mapping: UnitMapping) -> np.ndarray:
    return np.asarray(q_lu) * mapping.heat_flux_scale


def heat_flux_phys_to_lu(q_phys: np.ndarray, mapping: UnitMapping) -> np.ndarray:
    return np.asarray(q_phys) / mapping.heat_flux_scale
