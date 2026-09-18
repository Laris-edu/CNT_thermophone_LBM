"""Phase_5 multiharmonic joint fitter (contract Phase5_instruct_v1.2 §12.1/§12.2).

Model (contract §12.1):

    x(t) = a0 + a1*t + a2*t^2
         + sum_{n=1}^{N} [A_n cos(n Omega t) + B_n sin(n Omega t)],  default N=5

Complex-amplitude convention is the frozen project single source
(`phase3_interfaces/complex_amplitude.py`): ``x(t) = Re[x_hat_n exp(+i n Omega t)]``
with ``x_hat_n = A_n - 1j*B_n``. Harmonic design columns use the *raw* time axis,
so phases are referenced to ``t = 0`` of the caller's clock. Trend columns are
built on the internally scaled time ``tau = (t - t_mid)/t_half`` purely for
conditioning; coefficients are reported in both the scaled and the raw
``a0 + a1*t + a2*t^2`` basis. The reported condition number is that of the
scaled design matrix.

Required outputs (contract §12.1) and where they live here:

- complex amplitudes + covariance      -> ``MultiharmonicFit.x_hat`` / ``.covariance``
- design-matrix condition number       -> ``MultiharmonicFit.condition_number``
- residual spectrum                    -> ``residual_spectrum_fft`` / ``probe_amplitude``
- fit-window definition                -> ``MultiharmonicFit.window``
- detrend order                        -> ``MultiharmonicFit.detrend_order``
- multi-window sensitivity             -> ``multiwindow_sensitivity``
- synthetic-fixture leakage            -> ``leakage_fixture``

Detrend orders are pre-registered per protocol in ``PROTOCOL_DETREND``
(contract §12.2) and must not be chosen per-result (contract §0.4).
``U95`` helpers quantify *fit* uncertainty only (``U_95,fit`` in contract
§12.4); they are not the governing ``U_gov``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from phase3_interfaces.complex_amplitude import complex_amplitude

__all__ = [
    "PROTOCOL_DETREND",
    "FitWindow",
    "MultiharmonicFit",
    "WindowSensitivity",
    "fit_multiharmonic",
    "synthesize_multiharmonic",
    "residual_spectrum_fft",
    "probe_amplitude",
    "suffix_cycle_windows",
    "multiwindow_sensitivity",
    "leakage_fixture",
    "signed_pair_combination",
]

# Contract §12.2 pre-registration: protocol -> (main detrend order, sensitivity
# detrend order or None if no alternative is allowed). Quadratic trend is never
# a main fit for A1/A2a (it would absorb genuine slow physics).
PROTOCOL_DETREND: dict[str, tuple[int, int | None]] = {
    "A1": (0, 1),
    "A2a": (0, 1),
    "A2b": (1, 2),
    "G2_linear_fixture": (0, None),
}

_RANK_TOL = 1.0e-12


@dataclass(frozen=True)
class FitWindow:
    """Fit-window definition (contract §12.1 required output)."""

    t_start: float
    t_end: float
    n_samples: int
    n_cycles: float


@dataclass(frozen=True)
class MultiharmonicFit:
    """Result of one multiharmonic joint least-squares fit.

    Parameter ordering in ``covariance`` matches the design matrix:
    ``[tau^0 .. tau^detrend_order, A_1, B_1, ..., A_N, B_N]``.
    """

    omega_rad_s: float
    n_harmonics: int
    detrend_order: int
    x_hat: np.ndarray
    trend_coeffs_scaled: np.ndarray
    trend_coeffs_raw: np.ndarray
    trend_time_mid: float
    trend_time_half: float
    condition_number: float
    covariance: np.ndarray
    noise_sigma: float
    residual_rms: float
    dof: int
    window: FitWindow
    t: np.ndarray
    residual: np.ndarray

    # -- accessors ---------------------------------------------------------
    def _param_index(self, n: int) -> int:
        if not 1 <= n <= self.n_harmonics:
            raise ValueError(f"harmonic n={n} outside 1..{self.n_harmonics}")
        return (self.detrend_order + 1) + 2 * (n - 1)

    def harmonic(self, n: int) -> complex:
        self._param_index(n)
        return complex(self.x_hat[n - 1])

    def amplitude(self, n: int) -> float:
        return abs(self.harmonic(n))

    def phase_rad(self, n: int) -> float:
        x_hat = self.harmonic(n)
        return math.atan2(x_hat.imag, x_hat.real)

    def phase_deg(self, n: int) -> float:
        return math.degrees(self.phase_rad(n))

    def _ab_covariance(self, n: int) -> tuple[float, float, float]:
        k = self._param_index(n)
        return (
            float(self.covariance[k, k]),
            float(self.covariance[k + 1, k + 1]),
            float(self.covariance[k, k + 1]),
        )

    def amplitude_u95(self, n: int) -> float:
        """95% fit-only uncertainty of |x_hat_n| (linearized; ``U_95,fit``)."""

        var_a, var_b, cov_ab = self._ab_covariance(n)
        x_hat = self.harmonic(n)
        a, b = x_hat.real, -x_hat.imag  # x_hat = A - iB
        amp = abs(x_hat)
        if amp == 0.0:
            return 1.96 * math.sqrt(max(var_a, var_b, 0.0))
        var_amp = (a * a * var_a + b * b * var_b + 2.0 * a * b * cov_ab) / (amp * amp)
        return 1.96 * math.sqrt(max(var_amp, 0.0))

    def phase_u95_deg(self, n: int) -> float:
        """95% fit-only uncertainty of arg(x_hat_n) in degrees (linearized)."""

        var_a, var_b, cov_ab = self._ab_covariance(n)
        x_hat = self.harmonic(n)
        a, b = x_hat.real, -x_hat.imag
        amp2 = a * a + b * b
        if amp2 == 0.0:
            return float("inf")
        var_phase = (b * b * var_a + a * a * var_b - 2.0 * a * b * cov_ab) / (amp2 * amp2)
        return 1.96 * math.degrees(math.sqrt(max(var_phase, 0.0)))

    def leakage_relative(self, target: int = 1) -> dict[int, float]:
        """|x_hat_n| / |x_hat_target| for all fitted n != target."""

        ref = self.amplitude(target)
        if ref == 0.0:
            raise ValueError("target harmonic amplitude is zero; leakage undefined")
        return {
            n: self.amplitude(n) / ref
            for n in range(1, self.n_harmonics + 1)
            if n != target
        }

    def to_json_payload(self) -> dict:
        """Serializable payload for the per-run ``harmonic_fit.json`` (contract §16.1)."""

        harmonics = list(range(1, self.n_harmonics + 1))
        return {
            "model": "a0+a1*t+a2*t^2 + sum_n [A_n cos(n Omega t) + B_n sin(n Omega t)]",
            "phase_convention": "x(t)=Re[x_hat_n exp(+i n Omega t)]; x_hat_n=A_n-1j*B_n; t from caller clock",
            "omega_rad_s": self.omega_rad_s,
            "frequency_hz": self.omega_rad_s / (2.0 * math.pi),
            "harmonic_order_max": self.n_harmonics,
            "detrend_order": self.detrend_order,
            "fit_window": {
                "t_start": self.window.t_start,
                "t_end": self.window.t_end,
                "n_samples": self.window.n_samples,
            },
            "fit_cycles": self.window.n_cycles,
            "harmonic_fit_condition_number": self.condition_number,
            "x_hat_re": [self.harmonic(n).real for n in harmonics],
            "x_hat_im": [self.harmonic(n).imag for n in harmonics],
            "amplitude": [self.amplitude(n) for n in harmonics],
            "phase_deg": [self.phase_deg(n) for n in harmonics],
            "amplitude_u95_fit": [self.amplitude_u95(n) for n in harmonics],
            "phase_u95_fit_deg": [self.phase_u95_deg(n) for n in harmonics],
            "trend_coeffs_raw": [float(c) for c in self.trend_coeffs_raw],
            "trend_coeffs_scaled": [float(c) for c in self.trend_coeffs_scaled],
            "trend_time_mid": self.trend_time_mid,
            "trend_time_half": self.trend_time_half,
            "residual_rms": self.residual_rms,
            "noise_sigma": self.noise_sigma,
            "dof": self.dof,
        }


def _validate_series(t: np.ndarray, x: np.ndarray) -> None:
    if t.ndim != 1 or x.ndim != 1 or t.size != x.size:
        raise ValueError("t and x must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(x))):
        raise ValueError("t and x must be finite")
    if t.size < 2 or not np.all(np.diff(t) > 0.0):
        raise ValueError("t must be strictly increasing with at least 2 samples")


def _scaled_to_raw(coeffs_scaled: np.ndarray, t_mid: float, t_half: float) -> np.ndarray:
    raw = np.zeros(3, dtype=float)
    raw[0] = coeffs_scaled[0]
    if coeffs_scaled.size > 1:
        c1 = coeffs_scaled[1]
        raw[0] -= c1 * t_mid / t_half
        raw[1] += c1 / t_half
    if coeffs_scaled.size > 2:
        c2 = coeffs_scaled[2]
        h2 = t_half * t_half
        raw[0] += c2 * t_mid * t_mid / h2
        raw[1] -= 2.0 * c2 * t_mid / h2
        raw[2] += c2 / h2
    return raw


def fit_multiharmonic(
    t: np.ndarray,
    x: np.ndarray,
    omega_rad_s: float,
    *,
    n_harmonics: int = 5,
    detrend_order: int = 0,
) -> MultiharmonicFit:
    """Joint LS fit of trend + harmonics 1..N at ``omega_rad_s`` (contract §12.1)."""

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    _validate_series(t, x)
    if not (isinstance(n_harmonics, int) and n_harmonics >= 1):
        raise ValueError("n_harmonics must be an integer >= 1")
    if detrend_order not in (0, 1, 2):
        raise ValueError("detrend_order must be 0, 1 or 2 (contract §12.1 model)")
    if not (omega_rad_s > 0.0 and math.isfinite(omega_rad_s)):
        raise ValueError("omega_rad_s must be positive and finite")

    m = t.size
    ncols = (detrend_order + 1) + 2 * n_harmonics
    if m <= ncols:
        raise ValueError(f"need more than {ncols} samples for {ncols} parameters, got {m}")

    t_mid = 0.5 * (t[0] + t[-1])
    t_half = 0.5 * (t[-1] - t[0])
    tau = (t - t_mid) / t_half

    columns = [tau**j for j in range(detrend_order + 1)]
    for n in range(1, n_harmonics + 1):
        columns.append(np.cos(n * omega_rad_s * t))
        columns.append(np.sin(n * omega_rad_s * t))
    design = np.column_stack(columns)

    u_mat, s_vec, vt_mat = np.linalg.svd(design, full_matrices=False)
    if s_vec[-1] <= s_vec[0] * _RANK_TOL:
        raise ValueError("design matrix is rank-deficient or ill-conditioned for this window")
    condition_number = float(s_vec[0] / s_vec[-1])

    coeffs = vt_mat.T @ ((u_mat.T @ x) / s_vec)
    residual = x - design @ coeffs
    dof = m - ncols
    ssr = float(residual @ residual)
    sigma2 = ssr / dof
    covariance = (vt_mat.T * (1.0 / s_vec**2)) @ vt_mat * sigma2

    trend_scaled = np.array(coeffs[: detrend_order + 1], dtype=float)
    ab = np.asarray(coeffs[detrend_order + 1 :], dtype=float).reshape(n_harmonics, 2)
    x_hat = ab[:, 0] - 1j * ab[:, 1]

    window = FitWindow(
        t_start=float(t[0]),
        t_end=float(t[-1]),
        n_samples=int(m),
        n_cycles=float((t[-1] - t[0]) * omega_rad_s / (2.0 * math.pi)),
    )
    return MultiharmonicFit(
        omega_rad_s=float(omega_rad_s),
        n_harmonics=int(n_harmonics),
        detrend_order=int(detrend_order),
        x_hat=x_hat,
        trend_coeffs_scaled=trend_scaled,
        trend_coeffs_raw=_scaled_to_raw(trend_scaled, t_mid, t_half),
        trend_time_mid=float(t_mid),
        trend_time_half=float(t_half),
        condition_number=condition_number,
        covariance=covariance,
        noise_sigma=float(math.sqrt(sigma2)),
        residual_rms=float(math.sqrt(ssr / m)),
        dof=int(dof),
        window=window,
        t=t.copy(),
        residual=residual,
    )


def synthesize_multiharmonic(
    t: np.ndarray,
    omega_rad_s: float,
    x_hats: Mapping[int, complex],
    trend_raw: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Synthesize ``x(t)`` from complex amplitudes under the frozen convention."""

    t = np.asarray(t, dtype=float)
    x = np.zeros_like(t)
    for j, a_j in enumerate(trend_raw):
        if a_j != 0.0:
            x = x + a_j * t**j
    for n, x_hat in x_hats.items():
        if not (isinstance(n, int) and n >= 1):
            raise ValueError("harmonic keys must be integers >= 1")
        x = x + np.real(complex(x_hat) * np.exp(1j * n * omega_rad_s * t))
    return x


def residual_spectrum_fft(fit: MultiharmonicFit) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum of the fit residual (contract §12.1).

    Requires uniform sampling; raises otherwise (use :func:`probe_amplitude`
    for targeted probes on non-uniform grids). No extra taper is applied —
    interpret off-bin content with the usual rectangular-window caveat.
    """

    dt = np.diff(fit.t)
    if dt.size == 0:
        raise ValueError("residual spectrum needs at least 2 samples")
    dt_mean = float(np.mean(dt))
    if np.max(np.abs(dt - dt_mean)) > 1.0e-9 * dt_mean:
        raise ValueError("residual_spectrum_fft requires uniform sampling")
    m = fit.residual.size
    amplitude = np.abs(np.fft.rfft(fit.residual)) * (2.0 / m)
    amplitude[0] *= 0.5
    if m % 2 == 0:
        amplitude[-1] *= 0.5
    freq_hz = np.fft.rfftfreq(m, dt_mean)
    return freq_hz, amplitude


def probe_amplitude(t: np.ndarray, signal: np.ndarray, frequency_hz: float) -> complex:
    """Single-frequency LS probe delegating to the frozen convention source."""

    return complex_amplitude(t, signal, frequency_hz)


def suffix_cycle_windows(
    t: np.ndarray,
    omega_rad_s: float,
    cycle_counts: Sequence[float],
) -> list[tuple[float, float]]:
    """Windows ending at ``t[-1]`` spanning the last ``k`` carrier cycles each."""

    t = np.asarray(t, dtype=float)
    period = 2.0 * math.pi / float(omega_rad_s)
    span = t[-1] - t[0]
    windows: list[tuple[float, float]] = []
    for k in cycle_counts:
        length = float(k) * period
        if length <= 0.0 or length > span * (1.0 + 1.0e-12):
            raise ValueError(f"window of {k} cycles does not fit in the data span")
        windows.append((float(t[-1] - length), float(t[-1])))
    return windows


@dataclass(frozen=True)
class WindowSensitivity:
    """Multi-window sensitivity summary (contract §12.1 required output)."""

    windows: tuple[tuple[float, float], ...]
    reference_index: int
    amplitude_max_rel_dev: dict[int, float]
    phase_max_dev_deg: dict[int, float]
    fits: tuple[MultiharmonicFit, ...]


def _wrap_deg(delta: float) -> float:
    return abs((delta + 180.0) % 360.0 - 180.0)


def multiwindow_sensitivity(
    t: np.ndarray,
    x: np.ndarray,
    omega_rad_s: float,
    *,
    n_harmonics: int = 5,
    detrend_order: int = 0,
    windows: Sequence[tuple[float, float]],
    reference_index: int = 0,
    amplitude_floor: float = 0.0,
) -> WindowSensitivity:
    """Refit over each window; report per-harmonic max deviation vs the reference.

    ``amplitude_floor`` guards the relative-deviation denominator for harmonics
    whose reference amplitude is at the null-check floor (0 -> may yield inf).
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if not windows:
        raise ValueError("windows must be non-empty")
    tol = 1.0e-12 * max(abs(t[0]), abs(t[-1]), t[-1] - t[0])
    fits: list[MultiharmonicFit] = []
    for t_start, t_end in windows:
        mask = (t >= t_start - tol) & (t <= t_end + tol)
        fits.append(
            fit_multiharmonic(
                t[mask], x[mask], omega_rad_s,
                n_harmonics=n_harmonics, detrend_order=detrend_order,
            )
        )
    reference = fits[reference_index]
    amp_dev: dict[int, float] = {}
    phase_dev: dict[int, float] = {}
    for n in range(1, n_harmonics + 1):
        ref_amp = reference.amplitude(n)
        denom = max(ref_amp, amplitude_floor)
        deltas = [abs(f.amplitude(n) - ref_amp) for f in fits]
        if denom == 0.0:
            amp_dev[n] = 0.0 if max(deltas) == 0.0 else float("inf")
        else:
            amp_dev[n] = max(deltas) / denom
        phase_dev[n] = max(_wrap_deg(f.phase_deg(n) - reference.phase_deg(n)) for f in fits)
    return WindowSensitivity(
        windows=tuple((float(a), float(b)) for a, b in windows),
        reference_index=int(reference_index),
        amplitude_max_rel_dev=amp_dev,
        phase_max_dev_deg=phase_dev,
        fits=tuple(fits),
    )


def signed_pair_combination(
    t: np.ndarray,
    x_plus: np.ndarray,
    x_minus: np.ndarray,
    omega_rad_s: float,
    *,
    n_harmonics: int = 5,
    detrend_order: int = 0,
    threshold: float = 1.0e-8,
) -> dict:
    """Signed-pair (±drive) combination analysis — the practical
    "boundary + linearized interior" fixture (contract §6.1) for stacks that
    cannot be literally linearized.

    odd  = (x+ - x-)/2: even-order content (DC, 2f, 4f) cancels analytically;
           surviving 2f/4f measures the numerical even-harmonic floor of the
           operator chain (gate row: <=1e-8). 3f survives at O(eps^2) — genuine
           cubic physics plus floor; the caller decides its assertion by eps.
    even = (x+ + x-)/2: linear-in-drive content (including slow transients)
           cancels; its 2f relative to the odd 1f is the genuine physical
           second harmonic (sensitivity counter-check).

    Generalizes the antisymmetric-pair combination used by the reference
    fixture in the legacy JFM2024 repository (``reference/nonlinear_nsf_1d.py``)
    to arbitrary recorded QoI series (LBM wall flux, wall temperature, ...).
    """

    t = np.asarray(t, dtype=float)
    fit_odd = fit_multiharmonic(
        t, 0.5 * (np.asarray(x_plus, float) - np.asarray(x_minus, float)),
        omega_rad_s, n_harmonics=n_harmonics, detrend_order=detrend_order,
    )
    fit_even = fit_multiharmonic(
        t, 0.5 * (np.asarray(x_plus, float) + np.asarray(x_minus, float)),
        omega_rad_s, n_harmonics=n_harmonics, detrend_order=detrend_order,
    )
    leak_odd = fit_odd.leakage_relative(target=1)
    even_orders = [n for n in leak_odd if n % 2 == 0]
    max_even = max(leak_odd[n] for n in even_orders)
    physical_2f = fit_even.amplitude(2) / fit_odd.amplitude(1)
    return {
        "leakage_odd_combination": leak_odd,
        "max_even_leakage_odd_combination": float(max_even),
        "third_harmonic_odd_combination": float(leak_odd.get(3, float("nan"))),
        "physical_2f_rel_even_combination": float(physical_2f),
        "threshold": float(threshold),
        "even_floor_passed": bool(max_even <= threshold),
        "fits": {"odd": fit_odd, "even": fit_even},
    }


def leakage_fixture(
    omega_rad_s: float,
    *,
    n_harmonics: int = 5,
    n_cycles: int = 8,
    samples_per_cycle: int = 64,
    amplitude: float = 1.0,
    phase_deg: float = -30.0,
    target_harmonic: int = 1,
    dc_offset: float = 0.0,
    detrend_order: int = 0,
    threshold: float = 1.0e-8,
) -> dict:
    """Synthetic single-tone leakage fixture (contract §12.1; gate use e.g. §7.1/§8.2).

    Builds a clean single-tone record on a uniform, endpoint-excluded grid,
    runs the joint fit and reports the recovered non-target harmonic content
    relative to the target. Deterministic (no RNG). ``threshold`` transcribes
    the contract's non-target relative-leakage gate value 1e-8.
    """

    period = 2.0 * math.pi / float(omega_rad_s)
    m = int(n_cycles) * int(samples_per_cycle)
    t = np.arange(m, dtype=float) * (period / samples_per_cycle)
    x_hat_target = amplitude * np.exp(1j * math.radians(phase_deg))
    signal = synthesize_multiharmonic(
        t, omega_rad_s, {int(target_harmonic): complex(x_hat_target)},
        trend_raw=(dc_offset, 0.0, 0.0),
    )
    fit = fit_multiharmonic(
        t, signal, omega_rad_s, n_harmonics=n_harmonics, detrend_order=detrend_order
    )
    leakage = fit.leakage_relative(target=target_harmonic)
    max_leakage = max(leakage.values())
    target_error = abs(fit.harmonic(target_harmonic) - complex(x_hat_target)) / abs(x_hat_target)
    return {
        "target_harmonic": int(target_harmonic),
        "target_relative_error": float(target_error),
        "per_harmonic_relative_leakage": {int(n): float(v) for n, v in leakage.items()},
        "max_relative_leakage": float(max_leakage),
        "threshold": float(threshold),
        "passed": bool(max_leakage <= threshold),
        "fit_window": {
            "n_samples": fit.window.n_samples,
            "n_cycles": fit.window.n_cycles,
        },
        "condition_number": fit.condition_number,
    }
