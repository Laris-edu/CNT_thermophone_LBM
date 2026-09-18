"""Shared result containers for Phase 1 reference calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


def _complex_columns(prefix: str, value: complex) -> dict[str, float]:
    return {f"{prefix}_real": float(np.real(value)), f"{prefix}_imag": float(np.imag(value))}


def probe_tag(value: float) -> str:
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return text


@dataclass
class ReferenceResult:
    case_name: str
    level: str
    f_Hz: float
    Omega: float
    T0: float
    p0: float
    rho0: float
    gamma: float
    cp: float
    kg: float
    alpha0: float
    nu0: float
    Pr: float
    C_A: float
    beta0: float
    P_hat: complex | None
    T_s_hat_input: complex | None
    q_hat_input: complex | None
    m_T: complex
    delta_T: float
    delta_v: float
    Pi_C: float
    epsilon_P: float
    k_delta_T: float
    k_a: float
    T_s_hat: complex
    q_g_hat: complex
    probes_y: np.ndarray
    probes_y_over_delta_T: np.ndarray
    p_hat_probes: np.ndarray
    T_hat_probes: np.ndarray
    u_hat_probes: np.ndarray
    energy_residual_hat: complex = 0.0 + 0.0j
    energy_residual_rel: float = 0.0
    solver_name: str = "phase1_reference"
    N_y: int = 0
    L_y: float = math.nan
    dy_min: float = math.nan
    dy_max: float = math.nan
    bc_far_pressure: str = "outgoing"
    bc_far_temperature: str = "T_hat(L)=0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def probe_index(self, y_over_delta_T: float) -> int:
        return int(np.argmin(np.abs(self.probes_y_over_delta_T - y_over_delta_T)))

    def p_at(self, y_over_delta_T: float) -> complex:
        return complex(self.p_hat_probes[self.probe_index(y_over_delta_T)])

    def T_at(self, y_over_delta_T: float) -> complex:
        return complex(self.T_hat_probes[self.probe_index(y_over_delta_T)])

    def u_at(self, y_over_delta_T: float) -> complex:
        return complex(self.u_hat_probes[self.probe_index(y_over_delta_T)])

    def to_flat_dict(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {
            "case_name": self.case_name,
            "level": self.level,
            "f_Hz": self.f_Hz,
            "Omega": self.Omega,
            "T0": self.T0,
            "p0": self.p0,
            "rho0": self.rho0,
            "gamma": self.gamma,
            "cp": self.cp,
            "kg": self.kg,
            "alpha0": self.alpha0,
            "nu0": self.nu0,
            "Pr": self.Pr,
            "C_A": self.C_A,
            "beta0": self.beta0,
            "delta_T": self.delta_T,
            "delta_v": self.delta_v,
            "Pi_C": self.Pi_C,
            "epsilon_P": self.epsilon_P,
            "k_delta_T": self.k_delta_T,
            "k_a": self.k_a,
            "energy_residual_rel": self.energy_residual_rel,
            "solver_name": self.solver_name,
            "N_y": self.N_y,
            "L_y": self.L_y,
            "dy_min": self.dy_min,
            "dy_max": self.dy_max,
            "bc_far_pressure": self.bc_far_pressure,
            "bc_far_temperature": self.bc_far_temperature,
        }
        for name, value in (
            ("P_hat", self.P_hat),
            ("T_s_hat_input", self.T_s_hat_input),
            ("q_hat_input", self.q_hat_input),
            ("m_T", self.m_T),
            ("T_s_hat", self.T_s_hat),
            ("q_g_hat", self.q_g_hat),
            ("energy_residual_hat", self.energy_residual_hat),
        ):
            if value is None:
                row[f"{name}_real"] = math.nan
                row[f"{name}_imag"] = math.nan
            else:
                row.update(_complex_columns(name, complex(value)))
        for i, y_over in enumerate(self.probes_y_over_delta_T):
            tag = probe_tag(float(y_over))
            row[f"probe_y_{tag}_m"] = float(self.probes_y[i])
            row.update(_complex_columns(f"p_hat_y_{tag}", complex(self.p_hat_probes[i])))
            row.update(_complex_columns(f"T_hat_y_{tag}", complex(self.T_hat_probes[i])))
            row.update(_complex_columns(f"u_hat_y_{tag}", complex(self.u_hat_probes[i])))
        return row


@dataclass
class TimeSeriesResult:
    case_name: str
    level: str
    f_Hz: float | None
    Omega: float | None
    t: np.ndarray
    T_s: np.ndarray
    q_g: np.ndarray
    p_probe: np.ndarray
    P_in: np.ndarray
    energy_residual: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarmonicResult:
    x_hat: complex
    peak_abs: float
    rms_abs: float
    phase_rad: float
    phase_deg: float

