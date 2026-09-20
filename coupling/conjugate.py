"""一维膜气耦合入口；默认逐步守恒交换，旧预测校正路径仅供诊断。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from boundary.wall_dirichlet import (
    LEVEL_A_HEAT_FLUX_SIGN_CONVENTION,
    LEVEL_A_WALL_NORMAL_CONVENTION,
    apply_bottom_dirichlet_wall,
)
from boundary.wall_thermal_grad import make_bottom_grad_wall_callback
from boundary.wall_thermal_mass_neutral import make_symmetric_mass_neutral_wall_callback
from core.solver import GasSolver2D
from phase3_interfaces.heat_flux_extraction import (
    UPPER_GAS_WALL_NORMAL,
    extract_wall_heat_flux,
)
from phase3_interfaces.wall_state_contract import wall_state_from_temperature

from .drive import DriveSignal, evaluate_drive
from .energy_audit import EnergyAuditResult, audit_film_energy
from .film_ode import FilmOdeParams, film_rhs


BOTTOM_WALL_ROW = 0
LevelCScheme = Literal["conservative", "heun_picard1", "heun", "explicit_lagged"]


@dataclass(frozen=True)
class LevelCCouplingResult:
    """Sampled Level C coupling trajectory."""

    t_si: np.ndarray
    T_s_K: np.ndarray
    P_in_si: np.ndarray
    q_g_one_sided_si: np.ndarray
    dT_s_dt_K_s: np.ndarray
    theta_wall_lu: np.ndarray
    T_wall_K: np.ndarray
    pressure_probe_Pa: np.ndarray
    temperature_probe_K: np.ndarray
    energy_audit: EnergyAuditResult
    coupling_scheme: str
    picard_iterations: int
    predictor_corrector_delta_K: np.ndarray
    wall_temperature_error_K: np.ndarray
    finite: bool
    no_clipping_or_floor_used: bool = True
    wall_normal_convention: str = LEVEL_A_WALL_NORMAL_CONVENTION
    heat_flux_sign_convention: str = LEVEL_A_HEAT_FLUX_SIGN_CONVENTION
    wall_bc: str = "equilibrium_clamp"
    q_feedback_relax: float = 1.0
    # per-sample total gas mass sum(f) in LU (pure observation, added for the
    # Phase_5 G1b global-mass gate rows; None on results from older callers)
    mass_lu: np.ndarray | None = None
    q_extraction: str = "moment_row"
    # per-sample box-mean pressure in Pa (pure observation; feeds the G1b
    # frequency-domain energy-channel admittance readout)
    p_box_mean_Pa: np.ndarray | None = None
    # conservative 模式的 q[i] 是 (t[i-1], t[i]) 的区间平均，q[0]=0。
    q_sample_kind: str = "instantaneous"
    gas_energy_J_m2: np.ndarray | None = None
    wall_energy_increment_J_m2: np.ndarray | None = None
    input_work_increment_J_m2: np.ndarray | None = None
    leak_work_increment_J_m2: np.ndarray | None = None
    total_energy_audit: EnergyAuditResult | None = None

    @property
    def passed_smoke(self) -> bool:
        return bool(self.finite and self.energy_audit.passed and self.no_clipping_or_floor_used
                    and (self.total_energy_audit is None or self.total_energy_audit.passed))


def extract_bottom_wall_heat_flux_si(
    solver: GasSolver2D,
    *,
    row: int = BOTTOM_WALL_ROW,
    gas_offset: int = 1,
) -> float:
    """Extract the near-wall one-sided gas heat flux ``q_g''`` (SI, W/m^2).

    The conductive flux ``q_g''=-k_g dT/dy|0+`` is carried by the gas adjacent to the
    wall, not by the Dirichlet wall row: that row is clamped to an equilibrium state and
    therefore carries ~0 conductive flux. We follow the Phase_3 handoff convention
    (``scripts/phase2_phase3_handoff.py``): take the full-field conductive ``q_n(y)``
    profile and read the first interior gas row ``row + gas_offset``.
    """

    if row != BOTTOM_WALL_ROW:
        raise ValueError("P3-4 supports the bottom wall row only")
    if gas_offset < 1:
        raise ValueError("gas_offset must point into the gas (>=1), not at the clamped wall row")
    solver.get_macro()
    assert solver.f is not None and solver.g is not None
    q_field = extract_wall_heat_flux(
        solver.f,
        solver.g,
        wall_normal=UPPER_GAS_WALL_NORMAL,
        config=solver.config,
        return_physical=True,
    )
    q_n = np.mean(np.asarray(q_field, dtype=float), axis=1)  # per-row q_n(y)
    gas_row = min(row + gas_offset, solver.ny - 1)
    return float(q_n[gas_row])


def _pressure_probe_pa(solver: GasSolver2D, probe: tuple[int, int]) -> float:
    y, x = probe
    p_lu = float(solver.get_pressure_lu()[y, x])
    return float(p_lu * solver.mapping.pressure_scale)


def _temperature_probe_K(solver: GasSolver2D, probe: tuple[int, int]) -> float:
    y, x = probe
    theta_lu = float(solver.get_temperature_lu()[y, x])
    return float(theta_lu * solver.mapping.temperature_scale)


def _wall_temperature_error_K(solver: GasSolver2D, row: int, target_T_K: float) -> float:
    theta = solver.get_temperature_lu()[row]
    recovered_T = np.asarray(theta, dtype=float) * solver.mapping.temperature_scale
    return float(np.max(np.abs(recovered_T - float(target_T_K))))


def _wall_temperature_state(solver: GasSolver2D, row: int) -> tuple[float, float]:
    """Return the recovered mean wall temperature in lattice and SI units."""

    theta = np.asarray(solver.get_temperature_lu()[row], dtype=float)
    theta_mean = float(np.mean(theta))
    return theta_mean, theta_mean * float(solver.mapping.temperature_scale)


def _apply_wall_temperature(
    solver: GasSolver2D,
    *,
    T_wall_K: float,
    rho_policy: str,
    row: int,
) -> float:
    apply_bottom_dirichlet_wall(
        solver,
        T_wall_K=float(T_wall_K),
        rho_policy=rho_policy,  # type: ignore[arg-type]
        row=row,
    )
    return float(wall_state_from_temperature(T_wall_K, solver.config)["theta_wall_lu"])


def initialize_levelc_state(
    solver: GasSolver2D,
    *,
    T_initial_K: float,
    rho_policy: str = "pressure_preserving",
    row: int = BOTTOM_WALL_ROW,
) -> None:
    """Initialize a uniform gas state and clamp the Level C wall temperature."""

    solver.initialize_from_macro(
        solver.mapping.lattice.rho_ref_lu,
        np.zeros((solver.ny, solver.nx, 2), dtype=float),
        solver.mapping.theta_ref_lu,
    )
    _apply_wall_temperature(
        solver,
        T_wall_K=T_initial_K,
        rho_policy=rho_policy,
        row=row,
    )


def run_levelc_predictor_corrector(
    *,
    solver: GasSolver2D,
    params: FilmOdeParams,
    drive: DriveSignal | float,
    n_steps: int,
    dt_si: float | None = None,
    T_initial_K: float | None = None,
    rho_policy: str = "pressure_preserving",
    row: int = BOTTOM_WALL_ROW,
    scheme: LevelCScheme = "conservative",
    energy_tolerance: float = 1.0e-2,
    probe: tuple[int, int] | None = None,
    wall_bc: str | None = None,
    q_feedback_relax: float = 1.0,
    grad_extrap: str | None = None,
    q_extraction: str | None = None,
    q_feedback_scale: float = 1.0,
) -> LevelCCouplingResult:
    """运行膜气耦合；默认 conservative + mass_neutral + row1。

    保守路线用实际壁面能量交换联立求膜温，仅支持封闭对称双侧设置。
    q_g_one_sided_si[1:] 是前一区间平均热流，须在中点拟合，能量用
    sum(q[1:]*dt) 积分；不接受反馈增益、时间滤波或 row1 传导矩替代。

    heun、heun_picard1、explicit_lagged 是保留的旧诊断路线，不代表当前
    算例的推荐耦合。它们可选择 equilibrium_clamp、thermal_grad 或
    mass_neutral 壁，grad_extrap 选择 linear/row1 非平衡外推；
    q_extraction 可用 moment_row、energy_balance 或 hybrid_ac_dc。
    这些量分别是近壁传导矩、闭域能量估计和混合诊断反馈，不能混称
    保守壁面交换；旧路线的 q_feedback_scale/q_feedback_relax 也不能用来
    标定当前物理结果。历史分支与参数保留供诊断，不继承旧仓认证结论。
    """

    if wall_bc is None:
        wall_bc = "mass_neutral" if scheme == "conservative" else "equilibrium_clamp"
    if grad_extrap is None:
        grad_extrap = "row1" if scheme == "conservative" else "linear"
    if scheme == "conservative":
        if wall_bc != "mass_neutral" or row != 0:
            raise ValueError("conservative coupling requires the symmetric mass_neutral row-0 wall")
        if q_feedback_relax != 1.0 or q_feedback_scale != 1.0 or q_extraction not in (None, "wall_exchange"):
            raise ValueError("conservative coupling uses actual wall_exchange; no feedback gain or filter")
        from .conservative import run_conservative_conjugate
        return run_conservative_conjugate(solver=solver, params=params, drive=drive,
            n_steps=n_steps, dt_si=dt_si, T_initial_K=T_initial_K,
            energy_tolerance=energy_tolerance, probe=probe, grad_extrap=grad_extrap)
    q_extraction = "moment_row" if q_extraction is None else q_extraction
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if scheme not in {"heun_picard1", "heun", "explicit_lagged"}:
        raise ValueError(f"unsupported Level C scheme: {scheme}")
    if wall_bc not in {"equilibrium_clamp", "thermal_grad", "mass_neutral"}:
        raise ValueError(f"unsupported wall_bc: {wall_bc}")
    if grad_extrap not in {"linear", "row1"}:
        raise ValueError(f"unsupported grad_extrap: {grad_extrap}")
    if q_extraction not in {"moment_row", "energy_balance", "hybrid_ac_dc"}:
        raise ValueError(f"unsupported q_extraction: {q_extraction}")
    if not (q_feedback_scale > 0.0 and np.isfinite(q_feedback_scale)):
        raise ValueError("q_feedback_scale must be positive and finite")
    if not 0.0 < q_feedback_relax <= 1.0:
        raise ValueError("q_feedback_relax must be in (0, 1]")
    if row != BOTTOM_WALL_ROW:
        raise ValueError("P3-4 supports the bottom wall row only")

    gas_dt = float(solver.mapping.lattice.dt_s)
    dt = float(gas_dt if dt_si is None else dt_si)
    if dt <= 0.0:
        raise ValueError("dt_si must be positive")
    if not np.isclose(dt, gas_dt, rtol=1.0e-12, atol=0.0):
        raise ValueError(
            "dt_si must equal the LBM time step; gas subcycling is not implemented"
        )
    T0 = float(params.T_ref_K if T_initial_K is None else T_initial_K)
    probe_loc = probe or (min(max(row + 1, 0), solver.ny - 1), solver.nx // 2)

    initialize_levelc_state(
        solver,
        T_initial_K=T0,
        rho_policy=rho_policy,
        row=row,
    )

    t = np.arange(int(n_steps) + 1, dtype=float) * dt
    T_s = np.empty_like(t)
    P_in = np.empty_like(t)
    q_g = np.empty_like(t)
    dTdt = np.empty_like(t)
    theta_wall = np.empty_like(t)
    T_wall = np.empty_like(t)
    pressure_probe = np.empty_like(t)
    temperature_probe = np.empty_like(t)
    delta_pc = np.zeros_like(t)
    wall_error = np.empty_like(t)

    p_scale = float(solver.mapping.pressure_scale)
    gamma_g = float(solver.mapping.physical.gamma)
    energy_prefactor = solver.ny * float(solver.mapping.lattice.dx_m) / (2.0 * (gamma_g - 1.0))

    def _box_p_pa() -> float:
        return float(np.mean(solver.get_pressure_lu())) * p_scale

    p_box_prev = _box_p_pa() if q_extraction == "energy_balance" else 0.0
    if q_extraction == "hybrid_ac_dc":
        f_drive = float(getattr(drive, "frequency_hz", 0.0) or 0.0)
        if f_drive <= 0.0:
            raise ValueError("hybrid_ac_dc requires a drive with frequency_hz")
        n_period = max(int(round(1.0 / (f_drive * dt))), 8)
        hyb_buf_q = np.zeros(n_period)
        hyb_buf_p = np.zeros(n_period)
        hyb_sum_q = 0.0
        hyb_sum_p = 0.0
        hyb_p_init = _box_p_pa()
        hyb_g_dc = (float(getattr(solver.mapping.physical, "kg_W_mK", 0.0263))
                    / ((solver.ny / 2.0) * float(solver.mapping.lattice.dx_m)))
        hyb_step = 0
    p_box_series = np.empty_like(np.arange(int(n_steps) + 1, dtype=float))

    def _extract_q_raw(t_wall_now_K: float) -> float:
        nonlocal p_box_prev, hyb_sum_q, hyb_sum_p, hyb_step
        if q_extraction == "energy_balance":
            p_now = _box_p_pa()
            q_raw = energy_prefactor * (p_now - p_box_prev) / dt
            p_box_prev = p_now
            return q_feedback_scale * q_raw
        if q_extraction == "hybrid_ac_dc":
            q_m = q_feedback_scale * extract_bottom_wall_heat_flux_si(solver, row=row)
            p_now = _box_p_pa()
            idx = hyb_step % n_period
            # rolling one-period windows (O(1) per step)
            if hyb_step >= n_period:
                hyb_sum_q -= hyb_buf_q[idx]
                hyb_sum_p -= hyb_buf_p[idx]
            hyb_buf_q[idx] = q_m
            hyb_buf_p[idx] = p_now
            hyb_sum_q += q_m
            hyb_sum_p += p_now
            count = min(hyb_step + 1, n_period)
            m_q = hyb_sum_q / count
            p_bar = hyb_sum_p / count
            # v2 DC: physical pinning to the (slowly drifting) sealed-box mean
            # state — instantaneous in the film state (no delayed self-feedback),
            # smoothed only in the target; equilibrium independent of G_DC
            t_box_bar = T0 * p_bar / hyb_p_init
            q_dc = hyb_g_dc * (t_wall_now_K - t_box_bar)
            hyb_step += 1
            # v2.1: the de-DC operator is exact only with a FULL window; during
            # window fill the lagged mean lets the sign-inverted moment DC leak
            # through (measured: +307 K first-period ratchet, run 20260729T130108Z),
            # so the AC channel is gated off until the window is full (the film
            # couples through the DC pinning alone during period 1 = settle)
            ac = (q_m - m_q) if hyb_step > n_period else 0.0
            return ac + q_dc
        return q_feedback_scale * extract_bottom_wall_heat_flux_si(solver, row=row)

    mass_lu = np.empty_like(t)
    T_s[0] = T0
    P_in[0] = evaluate_drive(drive, float(t[0]))
    mass_lu[0] = float(np.sum(solver.f))
    p_box_series[0] = _box_p_pa()
    q_g[0] = (0.0 if q_extraction in ("energy_balance", "hybrid_ac_dc")
              else q_feedback_scale * extract_bottom_wall_heat_flux_si(solver, row=row))
    dTdt[0] = film_rhs(T_s[0], float(t[0]), params=params, drive=drive, q_g_one_sided_si=q_g[0])
    theta_wall[0], T_wall[0] = _wall_temperature_state(solver, row)
    pressure_probe[0] = _pressure_probe_pa(solver, probe_loc)
    temperature_probe[0] = _temperature_probe_K(solver, probe_loc)
    wall_error[0] = _wall_temperature_error_K(solver, row, T_s[0])

    picard_iterations = 1 if scheme == "heun_picard1" else 0

    def _theta_wall_lu(T_wall_K: float) -> float:
        return float(wall_state_from_temperature(T_wall_K, solver.config)["theta_wall_lu"])

    def _advance(T_wall_K: float) -> None:
        """Advance the gas one step while imposing the requested wall temperature."""
        if wall_bc == "thermal_grad":
            theta_w = _theta_wall_lu(T_wall_K)
            solver.step(1, boundary_callback=make_bottom_grad_wall_callback(
                theta_w, rho_policy=rho_policy, extrap=grad_extrap, fill_deep_links=False))
            return
        if wall_bc == "mass_neutral":
            theta_w = _theta_wall_lu(T_wall_K)
            solver.step(1, boundary_callback=make_symmetric_mass_neutral_wall_callback(
                theta_w, extrap=grad_extrap))
            return
        _apply_wall_temperature(solver, T_wall_K=T_wall_K, rho_policy=rho_policy, row=row)
        solver.step(1)
        _apply_wall_temperature(solver, T_wall_K=T_wall_K, rho_policy=rho_policy, row=row)

    def _reimpose(T_wall_K: float) -> None:
        """Picard-corrector wall re-imposition. For ``thermal_grad`` and ``mass_neutral``
        the wall is already imposed inside the streaming step; re-reconstructing row 0 here
        would retroactively alter the near-wall state (inconsistent with the predictor
        advance, and it detaches Level C from the isolated Level A admittance), so it is a
        no-op. For ``equilibrium_clamp`` it re-clamps row 0 (P3-4 behaviour)."""
        if wall_bc in ("thermal_grad", "mass_neutral"):
            return
        _apply_wall_temperature(solver, T_wall_K=T_wall_K, rho_policy=rho_policy, row=row)

    q_fb = float(q_g[0])  # under-relaxed q_g fed to the film ODE

    for i in range(int(n_steps)):
        t_n = float(t[i])
        t_np1 = float(t[i + 1])
        T_n = float(T_s[i])
        rhs_n = film_rhs(T_n, t_n, params=params, drive=drive, q_g_one_sided_si=q_fb)

        if scheme == "explicit_lagged":
            T_next = T_n + dt * rhs_n
            _advance(T_next)
            q_raw = _extract_q_raw(T_next)
            q_fb = (1.0 - q_feedback_relax) * q_fb + q_feedback_relax * q_raw
        else:
            T_predict = T_n + dt * rhs_n
            _advance(T_predict)
            q_raw = _extract_q_raw(T_predict)
            q_fb = (1.0 - q_feedback_relax) * q_fb + q_feedback_relax * q_raw
            rhs_end = film_rhs(T_predict, t_np1, params=params, drive=drive, q_g_one_sided_si=q_fb)
            T_next = T_n + 0.5 * dt * (rhs_n + rhs_end)
            delta_pc[i + 1] = T_next - T_predict
            for _ in range(picard_iterations):
                _reimpose(T_next)
                rhs_end = film_rhs(T_next, t_np1, params=params, drive=drive, q_g_one_sided_si=q_fb)
                T_next = T_n + 0.5 * dt * (rhs_n + rhs_end)
            _reimpose(T_next)
        # Record the q_g the film ODE actually integrated (q_fb): for equilibrium_clamp with
        # q_feedback_relax=1.0 this equals the raw extraction (unchanged); for thermal_grad it
        # keeps the integrated energy audit self-consistent with the under-relaxed feedback.
        q_next = q_fb

        T_s[i + 1] = T_next
        P_in[i + 1] = evaluate_drive(drive, t_np1)
        q_g[i + 1] = q_next
        dTdt[i + 1] = (T_s[i + 1] - T_s[i]) / dt
        theta_wall[i + 1], T_wall[i + 1] = _wall_temperature_state(solver, row)
        pressure_probe[i + 1] = _pressure_probe_pa(solver, probe_loc)
        temperature_probe[i + 1] = _temperature_probe_K(solver, probe_loc)
        wall_error[i + 1] = _wall_temperature_error_K(solver, row, T_s[i + 1])
        mass_lu[i + 1] = float(np.sum(solver.f))
        p_box_series[i + 1] = _box_p_pa()

    audit = audit_film_energy(
        t_si=t,
        P_in_si=P_in,
        q_g_one_sided_si=q_g,
        T_s_K=T_s,
        params=params,
        tolerance=energy_tolerance,
    )
    finite = bool(
        np.isfinite(T_s).all()
        and np.isfinite(P_in).all()
        and np.isfinite(q_g).all()
        and np.isfinite(pressure_probe).all()
        and np.isfinite(temperature_probe).all()
        and np.isfinite(wall_error).all()
        and np.isfinite(solver.f).all()
        and np.isfinite(solver.g).all()
    )
    return LevelCCouplingResult(
        t_si=t,
        T_s_K=T_s,
        P_in_si=P_in,
        q_g_one_sided_si=q_g,
        dT_s_dt_K_s=dTdt,
        theta_wall_lu=theta_wall,
        T_wall_K=T_wall,
        pressure_probe_Pa=pressure_probe,
        temperature_probe_K=temperature_probe,
        energy_audit=audit,
        coupling_scheme=scheme,
        picard_iterations=picard_iterations,
        predictor_corrector_delta_K=delta_pc,
        wall_temperature_error_K=wall_error,
        finite=finite,
        wall_bc=wall_bc,
        q_feedback_relax=q_feedback_relax,
        mass_lu=mass_lu,
        q_extraction=q_extraction,
        p_box_mean_Pa=p_box_series,
    )


__all__ = [
    "BOTTOM_WALL_ROW",
    "LevelCCouplingResult",
    "extract_bottom_wall_heat_flux_si",
    "initialize_levelc_state",
    "run_levelc_predictor_corrector",
]
