"""P4-4: 2D frequency-domain Kirchhoff / Helmholtz extrapolation kernel (contract section 9).

Independent of the LBM stack: pure Helmholtz machinery, anchored by manufactured fixtures
(verification/test_phase4_kirchhoff.py, driven by configs/phase4_kirchhoff_fixture.yaml via
scripts/phase4_kirchhoff_verification.py). Contract gates: manufactured amplitude error < 2%,
phase error < 2 deg. The prefactor and Hankel kind are FIXED here and by those fixtures only
-- never re-tuned against end-to-end thermoacoustic results (contract sections 2.4 / 9.3).

GREEN CONVENTION (the one-time resolution the P4-0 freeze demanded): the project's frozen
complex convention is x(t) = Re[x_hat exp(+i Omega t)] (P3-0). Under exp(+i Omega t) an
OUTGOING 2D cylindrical wave decays with phase exp(-ikR), i.e. H_0^(2)(kR); the free-space
Green function solving (nabla^2 + k^2) G = -delta is

    G(R) = (-i/4) H_0^(2)(kR)

which is the complex conjugate of the exp(-i omega t) textbook kernel (i/4) H_0^(1)(kR). The
plan-document shorthand `G ∝ H_0^(1)(kR)` / metadata name `hankel1_2d_outgoing` (contract
section 2.4) is that textbook form written without a time convention; under THIS project's
frozen exp(+i Omega t) it maps to the H_0^(2) kernel above. The cylindrical/phase fixtures pin
this pairing with discriminating power (the H_0^(1) kernel under this convention is an
INCOMING wave and fails reconstruction at O(1) -- kept selectable only as a counterexample).

INTEGRAL AND NORMAL CONVENTION (Green's second identity, observation domain above the
surface, radiation condition closing the far boundary for outgoing p and G):

    p_hat(x_obs) = Integral_S [ p_hat(y) dG/dn(y) - G(y) dp_hat/dn(y) ] dS(y)

with n the unit normal pointing FROM the source region INTO the observation domain (for the
top control surface: n = +e_y). This is exactly the contract section 2.4 shorthand. The
gradient channel accepts dp_hat/dn directly, or via the momentum relation from the normal
velocity amplitude (dpdn_from_velocity): under exp(+i Omega t),
i Omega rho0 v_hat = -grad p_hat  =>  dp_hat/dn = -i Omega rho0 v_hat_n."""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1, hankel2

# Contract section 2.4: the convention is frozen INTO THE MODULE as metadata.
KIRCHHOFF_METADATA = {
    "green_function": "hankel2_2d_outgoing",
    "green_function_contract_shorthand": (
        "hankel1_2d_outgoing (plan shorthand, no time convention); under the frozen "
        "exp(+i Omega t) the outgoing kernel is the conjugate form (-i/4) H0^(2)(kR)"
    ),
    "complex_convention": "Re[x_hat exp(i Omega t)]",
    "time_dependence": "exp(+i Omega t)",
    "prefactor": "(-i/4)",
    "prefactor_anchor": (
        "manufactured cylindrical-wave fixture (verification/test_phase4_kirchhoff.py); "
        "never tuned against end-to-end thermoacoustic results"
    ),
    "normal_convention": "n points from the source region into the observation domain",
}

GREEN_OUTGOING = "hankel2_2d_outgoing"
# Counterexample kernel: under exp(+i Omega t) this is an INCOMING wave. Selectable only so
# the convention fixture can demonstrate discriminating power; never a production choice.
GREEN_WRONG_TIME_CONVENTION = "hankel1_2d_wrong_time_convention_counterexample"


def dpdn_from_velocity(
    v_hat_n_m_s: np.ndarray | complex,
    *,
    omega_rad_s: float,
    rho0_kg_m3: float,
) -> np.ndarray | complex:
    """Normal pressure gradient from normal velocity: dp/dn = -i Omega rho0 v_n.

    Momentum equation under exp(+i Omega t). This is the velocity input channel of the
    control surface (contract section 6 records both channels and their difference)."""

    return -1j * omega_rad_s * rho0_kg_m3 * np.asarray(v_hat_n_m_s)


def kirchhoff_2d_frequency(
    *,
    surface_x_m: np.ndarray,
    surface_y_m: np.ndarray,
    normal_x: np.ndarray,
    normal_y: np.ndarray,
    ds_m: np.ndarray,
    p_hat_Pa: np.ndarray,
    dpdn_hat_Pa_m: np.ndarray,
    observer_x_m: np.ndarray,
    observer_y_m: np.ndarray,
    omega_rad_s: float,
    c0_m_s: float,
    green_convention: str,
) -> np.ndarray:
    """Return farfield p_hat_Pa at observer points (contract section 9.2 API).

    Discretized p(x_o) = sum_j [ p_j dG/dn_j - G_j (dp/dn)_j ] ds_j over surface samples."""

    if green_convention == GREEN_OUTGOING:
        h0, h1, pref = hankel2, hankel2, -0.25j
    elif green_convention == GREEN_WRONG_TIME_CONVENTION:
        h0, h1, pref = hankel1, hankel1, +0.25j
    else:
        raise ValueError(
            f"unknown green_convention: {green_convention!r} "
            f"(use {GREEN_OUTGOING!r}; the hankel1 form is a fixture counterexample only)"
        )

    sx = np.asarray(surface_x_m, dtype=float).ravel()
    sy = np.asarray(surface_y_m, dtype=float).ravel()
    nx = np.asarray(normal_x, dtype=float).ravel()
    ny = np.asarray(normal_y, dtype=float).ravel()
    ds = np.asarray(ds_m, dtype=float).ravel()
    p = np.asarray(p_hat_Pa, dtype=complex).ravel()
    dpdn = np.asarray(dpdn_hat_Pa_m, dtype=complex).ravel()
    ox = np.asarray(observer_x_m, dtype=float).ravel()
    oy = np.asarray(observer_y_m, dtype=float).ravel()
    if not (sx.size == sy.size == nx.size == ny.size == ds.size == p.size == dpdn.size):
        raise ValueError("surface arrays must share one length")

    k = float(omega_rad_s) / float(c0_m_s)
    rx = ox[:, None] - sx[None, :]
    ry = oy[:, None] - sy[None, :]
    r = np.sqrt(rx * rx + ry * ry)
    if np.any(r <= 0.0):
        raise ValueError("observer coincides with a surface sample")
    g = pref * h0(0, k * r)
    # dG/dn(y) = dG/dR * (grad_y R . n) with grad_y R = -R_hat: dG/dR = -pref*k*H1(kR)
    # => dG/dn = pref * k * H1(kR) * (R_hat . n)
    rhat_dot_n = (rx * nx[None, :] + ry * ny[None, :]) / r
    dgdn = pref * k * h1(1, k * r) * rhat_dot_n
    return (p[None, :] * dgdn - g * dpdn[None, :]) @ ds
