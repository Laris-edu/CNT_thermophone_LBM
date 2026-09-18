"""Frozen physical constants and scales for Phase 1 reference models."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


DEFAULT_PROBES_OVER_DELTA_T = (0.0, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0)


@dataclass(frozen=True)
class PhysicalParams:
    """SI-unit physical parameters frozen by Phase 0."""

    T0: float = 300.0
    p0: float = 101_325.0
    rho0: float = 1.177
    c0: float = 347.0
    gamma: float = 1.4
    cp: float = 1005.0
    kg: float = 0.0263
    nu0: float = 1.57e-5
    mu_bulk: float = 0.0
    C_A: float = 7.0e-4
    beta0: float = 0.0
    film_half_width: float = 5.0e-3

    @property
    def alpha0(self) -> float:
        return self.kg / (self.rho0 * self.cp)

    @property
    def Pr(self) -> float:
        return self.nu0 / self.alpha0

    @property
    def R(self) -> float:
        return self.p0 / (self.rho0 * self.T0)

    @property
    def cv(self) -> float:
        return self.cp / self.gamma

    @property
    def mu(self) -> float:
        return self.rho0 * self.nu0

    @property
    def mu_L(self) -> float:
        return 4.0 * self.mu / 3.0 + self.mu_bulk


def default_params(**overrides: float) -> PhysicalParams:
    """Return Phase 0 defaults, optionally replacing selected fields."""

    params = PhysicalParams()
    if overrides:
        params = replace(params, **overrides)
    return params


def omega_from_frequency(f_hz: float) -> float:
    return 2.0 * math.pi * float(f_hz)


def thermal_scales(
    f_hz: float,
    params: PhysicalParams | None = None,
    *,
    P_hat: float = 1000.0,
    C_A: float | None = None,
) -> dict[str, float]:
    """Return the derived Phase 1 scales for a frequency."""

    params = params or default_params()
    C_A = params.C_A if C_A is None else C_A
    Omega = omega_from_frequency(f_hz)
    delta_T = math.sqrt(2.0 * params.alpha0 / Omega)
    delta_v = math.sqrt(2.0 * params.nu0 / Omega)
    k_ac = Omega / params.c0
    return {
        "f_Hz": float(f_hz),
        "Omega": Omega,
        "alpha0": params.alpha0,
        "delta_T": delta_T,
        "delta_v": delta_v,
        "Pr": params.Pr,
        "Pi_C": Omega * C_A * delta_T / (2.0 * params.kg),
        "epsilon_P": P_hat * delta_T / (2.0 * params.kg * params.T0),
        "k_delta_T": k_ac * delta_T,
        "k_a": k_ac * params.film_half_width,
    }


def relative_error(value: complex | float, reference: complex | float) -> float:
    scale = max(abs(reference), 1e-300)
    return abs(value - reference) / scale

