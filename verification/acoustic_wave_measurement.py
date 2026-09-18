"""Real periodic acoustic-wave sound-speed diagnostics for Phase 2 P2-6."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.solver import GasSolver2D
from verification.shear_wave_measurement import _fit_decay


DEFAULT_DIRECTIONS = ("x", "y")
DEFAULT_SOUND_SPEED_TOLERANCE = 0.02
DEFAULT_GAMMA_TOLERANCE = 0.02
DEFAULT_DIRECTION_TOLERANCE = 0.02
ACOUSTIC_ATTENUATION_TARGET_POLICY = "MATCHED_LINEARIZED_NSF_D2_BULK_ZERO_CP_ALPHA"
ACOUSTIC_ATTENUATION_STATUS = "DIAGNOSTIC_ONLY_MATCHED_NSF_TARGET_DERIVED_GO_RISK"


@dataclass(frozen=True)
class AcousticWaveSettings:
    nx: int = 64
    ny: int = 64
    steps: int = 240
    sample_interval: int = 1
    mode_index: int = 1
    amplitude: float = 1.0e-6
    fit_start: int = 10
    fit_stop: int | None = None
    directions: tuple[str, ...] = DEFAULT_DIRECTIONS
    sound_speed_tolerance: float = DEFAULT_SOUND_SPEED_TOLERANCE
    gamma_tolerance: float = DEFAULT_GAMMA_TOLERANCE
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE
    background_velocity_lu: tuple[float, float] = (0.0, 0.0)
    # Phase_5 G0 (contract §5.1): uniform non-reference background on the SAME
    # frozen mapping. None -> theta_ref (behaviour identical to Phase_2).
    background_theta_lu: float | None = None
    background_path: str = "isobaric"  # "isobaric" | "equal_density"


def _settings_from_config(config: dict[str, Any]) -> AcousticWaveSettings:
    p2 = dict(config.get("p2_06_acoustic_wave", {}) or {})
    initial = dict(config.get("initial_condition", {}) or {})
    if initial.get("type") == "acoustic_eigenmode":
        p2.setdefault("amplitude", initial.get("amplitude"))
        p2.setdefault("mode_index", initial.get("wavenumber_mode"))
        if "directions" not in p2 and "direction" in initial:
            p2["directions"] = [initial["direction"]]

    directions = p2.get("directions", DEFAULT_DIRECTIONS)
    if isinstance(directions, str):
        directions = [directions]
    directions_tuple = tuple(str(item) for item in directions) or DEFAULT_DIRECTIONS

    return AcousticWaveSettings(
        nx=int(p2.get("nx", 64)),
        ny=int(p2.get("ny", 64)),
        steps=int(p2.get("steps", 240)),
        sample_interval=int(p2.get("sample_interval", 1)),
        mode_index=int(p2.get("mode_index", p2.get("wavenumber_mode", 1))),
        amplitude=float(p2.get("amplitude", 1.0e-6)),
        fit_start=int(p2.get("fit_start", 10)),
        fit_stop=None if p2.get("fit_stop", None) is None else int(p2["fit_stop"]),
        directions=directions_tuple,
        sound_speed_tolerance=float(p2.get("sound_speed_tolerance", DEFAULT_SOUND_SPEED_TOLERANCE)),
        gamma_tolerance=float(p2.get("gamma_tolerance", DEFAULT_GAMMA_TOLERANCE)),
        direction_tolerance=float(p2.get("direction_tolerance", DEFAULT_DIRECTION_TOLERANCE)),
        background_velocity_lu=_velocity_pair(p2.get("background_velocity_lu", (0.0, 0.0))),
        background_theta_lu=(
            None if p2.get("background_theta_lu") is None else float(p2["background_theta_lu"])
        ),
        background_path=str(p2.get("background_path", "isobaric")),
    )


def _background_state(mapping: Any, settings: Any) -> tuple[float, float]:
    """Uniform background (theta_b, rho_b) in LU on the frozen mapping (G0 §5.1)."""

    theta_ref = float(mapping.theta_ref_lu)
    rho_ref = float(mapping.lattice.rho_ref_lu)
    theta_b = theta_ref if settings.background_theta_lu is None else float(settings.background_theta_lu)
    if theta_b <= 0.0:
        raise ValueError("background_theta_lu must be positive")
    if settings.background_path == "isobaric":
        rho_b = rho_ref * theta_ref / theta_b
    elif settings.background_path == "equal_density":
        rho_b = rho_ref
    else:
        raise ValueError(f"unknown background_path: {settings.background_path}")
    return theta_b, rho_b


def _velocity_pair(value: Any) -> tuple[float, float]:
    if isinstance(value, int | float):
        return (float(value), 0.0)
    if isinstance(value, dict):
        return (float(value.get("ux", 0.0)), float(value.get("uy", 0.0)))
    items = list(value)
    if len(items) != 2:
        raise ValueError("background_velocity_lu must contain ux and uy")
    return (float(items[0]), float(items[1]))


def _simulation_config(config: dict[str, Any], settings: AcousticWaveSettings, direction: str) -> dict[str, Any]:
    sim_config = deepcopy(config)
    sim_config["numerics"] = dict(sim_config.get("numerics", {}) or {})
    sim_config["numerics"]["nx"] = settings.nx
    sim_config["numerics"]["ny"] = settings.ny
    sim_config["case"] = dict(sim_config.get("case", {}) or {})
    sim_config["case"]["name"] = f"p2_06_acoustic_wave_{direction}"
    return sim_config


def _mesh(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    return np.meshgrid(x, y)


def _direction_phase_and_unit(
    direction: str,
    ny: int,
    nx: int,
    mode_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_grid, y_grid = _mesh(ny, nx)
    if direction == "x":
        kx = 2.0 * np.pi * mode_index / nx
        phase = kx * x_grid
        unit = np.array([1.0, 0.0])
        k_mag = kx
    elif direction == "y":
        ky = 2.0 * np.pi * mode_index / ny
        phase = ky * y_grid
        unit = np.array([0.0, 1.0])
        k_mag = ky
    elif direction == "diagonal":
        kx = 2.0 * np.pi * mode_index / nx
        ky = 2.0 * np.pi * mode_index / ny
        k_mag = float(np.sqrt(kx * kx + ky * ky))
        phase = kx * x_grid + ky * y_grid
        unit = np.array([kx, ky], dtype=float) / k_mag
    else:
        raise ValueError("direction must be x, y, or diagonal")
    return phase, unit, float(k_mag)


def _modal_amplitude_2d(field: np.ndarray, phase: np.ndarray) -> complex:
    arr = np.asarray(field, dtype=float)
    centered = arr - np.mean(arr)
    return complex((2.0 / centered.size) * np.sum(centered * np.exp(-1j * phase)))


def _initialize_acoustic_wave(
    solver: GasSolver2D,
    settings: AcousticWaveSettings,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    phase, unit, k_mag = _direction_phase_and_unit(direction, solver.ny, solver.nx, settings.mode_index)
    mapping = solver.mapping
    gamma = mapping.physical.gamma
    theta0, rho0 = _background_state(mapping, settings)
    c_s = float(np.sqrt(gamma * theta0))

    rho = rho0 * (1.0 + settings.amplitude * np.sin(phase))
    theta = theta0 * (rho / rho0) ** (gamma - 1.0)
    u = np.broadcast_to(np.asarray(settings.background_velocity_lu, dtype=float), (solver.ny, solver.nx, 2)).copy()
    u += (c_s * settings.amplitude * np.sin(phase))[..., None] * unit[None, None, :]
    solver.initialize_from_macro(rho, u, theta)
    return phase, unit, k_mag


def _prony_decay_rate(
    times: np.ndarray,
    amplitudes: np.ndarray,
    start: int,
    stop: int | None,
    order: int = 3,
) -> float:
    """Window-free acoustic damping via a Prony / linear-prediction fit.

    A weakly damped acoustic standing/travelling wave is a sum of forward +
    backward eigenmodes; the log|p'| slope is biased by their beating at short
    windows. Decomposing the complex pressure modal amplitude into a few complex
    exponentials and returning the damping of the DOMINANT mode is window-free
    and equals the exact one-step modal eigenvalue (the recommended caliber). On
    too-few samples or a degenerate fit it returns NaN (caller falls back to the
    log|p'| slope).
    """
    finite = np.isfinite(times) & np.isfinite(np.abs(amplitudes))
    stop_value = (int(np.max(times[finite])) if np.any(finite) else start) if stop is None else stop
    mask = finite & (times >= start) & (times <= stop_value)
    x = np.asarray(amplitudes[mask], dtype=complex)
    n = x.size
    if n < 2 * order + 2:
        return np.nan
    try:
        rows = n - order
        matrix = np.empty((rows, order), dtype=complex)
        rhs = np.empty(rows, dtype=complex)
        for i in range(rows):
            matrix[i, :] = x[i:i + order][::-1]
            rhs[i] = -x[i + order]
        coeffs = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        roots = np.roots(np.concatenate(([1.0], coeffs)))
        roots = roots[np.isfinite(roots) & (np.abs(roots) > 0.0)]
        if roots.size == 0:
            return np.nan
        s = np.log(roots)  # per-step complex rate
        tt = (times[mask] - times[mask][0]).astype(float)
        vander = np.exp(np.outer(tt, s))
        amp = np.linalg.lstsq(vander, x, rcond=None)[0]
        dominant = int(np.argmax(np.abs(amp)))
        return -float(s[dominant].real)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def _fit_phase(
    times: np.ndarray,
    amplitudes: np.ndarray,
    start: int,
    stop: int | None,
) -> dict[str, Any]:
    finite = np.isfinite(times) & np.isfinite(np.abs(amplitudes))
    if stop is None:
        stop_value = int(np.max(times[finite])) if np.any(finite) else start
    else:
        stop_value = stop
    mask = finite & (times >= start) & (times <= stop_value)
    if np.count_nonzero(mask) < 3:
        return {
            "phase_slope": np.nan,
            "phase_intercept": np.nan,
            "phase_residual_norm": np.nan,
            "fitting_window": [int(start), int(stop_value)],
            "fit_sample_count": int(np.count_nonzero(mask)),
        }

    t_fit = times[mask]
    phase = np.unwrap(np.angle(amplitudes[mask]))
    matrix = np.column_stack((np.ones_like(t_fit), t_fit))
    coeffs, *_ = np.linalg.lstsq(matrix, phase, rcond=None)
    intercept, slope = coeffs
    residual = phase - (intercept + slope * t_fit)
    return {
        "phase_slope": float(slope),
        "phase_intercept": float(intercept),
        "phase_residual_norm": float(np.sqrt(np.mean(residual * residual))),
        "fitting_window": [int(np.min(t_fit)), int(np.max(t_fit))],
        "fit_sample_count": int(t_fit.size),
    }


def _matched_nsf_acoustic_attenuation_coeff_lu(mapping: Any) -> float:
    """Return the small-k linearized NSF acoustic amplitude damping coefficient.

    ``alpha_lu`` uses the conductive heat-flux convention from P2-5,
    alpha=k/(rho cp). The D2Q37 conductive heat-flux moment/export factors
    define how raw central energy flux is reported as Fourier-law ``q_lu``;
    they do not multiply the continuum NSF target again.
    """

    coeff = (
        0.5 * (mapping.physical.gamma - 1.0) * mapping.alpha_lu
        + ((mapping.lattice.D - 1.0) / mapping.lattice.D) * mapping.nu_lu
        + 0.5 * mapping.nu_b_lu
    )
    return float(coeff)


def _reference_acoustic_attenuation_rate(solver: GasSolver2D, k_mag: float) -> float:
    mapping = solver.mapping
    coeff = _matched_nsf_acoustic_attenuation_coeff_lu(mapping)
    return float(coeff * k_mag * k_mag)


def measure_acoustic_wave_direction(
    config: dict[str, Any],
    direction: str,
    settings: AcousticWaveSettings,
) -> dict[str, Any]:
    if settings.amplitude > 1.0e-5:
        raise ValueError("P2-6 acoustic amplitude must satisfy rho'/rho0 <= 1e-5")
    if settings.sample_interval <= 0:
        raise ValueError("sample_interval must be positive")

    solver = GasSolver2D(_simulation_config(config, settings, direction))
    phase, unit, k_mag = _initialize_acoustic_wave(solver, settings, direction)
    mapping = solver.mapping
    theta0, rho0 = _background_state(mapping, settings)
    p0 = rho0 * theta0
    c_target = float(np.sqrt(mapping.physical.gamma * theta0))
    background_advection = float(np.dot(np.asarray(settings.background_velocity_lu, dtype=float), unit))

    times: list[int] = []
    pressure_amplitudes: list[complex] = []
    density_amplitudes: list[complex] = []
    theta_amplitudes: list[complex] = []
    velocity_amplitudes: list[complex] = []
    first_invalid_step: int | None = None
    nan_detected = False
    negative_theta_detected = False
    min_theta = np.inf
    max_theta = -np.inf
    max_mach = 0.0
    max_mean_pressure_drift = 0.0

    for step in range(settings.steps + 1):
        with np.errstate(all="ignore"):
            macro = solver.get_macro()
        finite = bool(np.isfinite(macro.rho).all() and np.isfinite(macro.u).all() and np.isfinite(macro.theta).all())
        nan_detected = nan_detected or not finite
        theta_min = float(np.nanmin(macro.theta)) if macro.theta.size else np.nan
        theta_max = float(np.nanmax(macro.theta)) if macro.theta.size else np.nan
        min_theta = min(min_theta, theta_min)
        max_theta = max(max_theta, theta_max)
        max_mach = max(max_mach, float(np.nanmax(macro.mach)))
        max_mean_pressure_drift = max(max_mean_pressure_drift, float(abs(np.nanmean(macro.p) - p0)))
        if (not finite) or theta_min <= 0.0:
            first_invalid_step = step
            negative_theta_detected = negative_theta_detected or bool(theta_min <= 0.0)
            break

        if step % settings.sample_interval == 0:
            u_normal = np.einsum("...i,i->...", macro.u, unit)
            times.append(step)
            pressure_amplitudes.append(_modal_amplitude_2d(macro.p - p0, phase))
            density_amplitudes.append(_modal_amplitude_2d(macro.rho - rho0, phase))
            theta_amplitudes.append(_modal_amplitude_2d(macro.theta - theta0, phase))
            velocity_amplitudes.append(_modal_amplitude_2d(u_normal - background_advection, phase))
        if step < settings.steps:
            with np.errstate(all="ignore"):
                solver.step()

    time_array = np.asarray(times, dtype=float)
    pressure_array = np.asarray(pressure_amplitudes, dtype=complex)
    phase_fit = _fit_phase(time_array, pressure_array, settings.fit_start, settings.fit_stop)
    decay_fit = _fit_decay(time_array, pressure_array, settings.fit_start, settings.fit_stop)

    phase_slope = phase_fit["phase_slope"]
    lab_phase_speed = -phase_slope / k_mag if np.isfinite(phase_slope) and k_mag > 0.0 else np.nan
    intrinsic_phase_speed = lab_phase_speed - background_advection if np.isfinite(lab_phase_speed) else np.nan
    sound_speed_measured = abs(intrinsic_phase_speed) if np.isfinite(intrinsic_phase_speed) else np.nan
    gamma_measured = sound_speed_measured**2 / theta0 if np.isfinite(sound_speed_measured) else np.nan
    sound_speed_relative_error = (
        abs(sound_speed_measured / c_target - 1.0) if np.isfinite(sound_speed_measured) else np.nan
    )
    gamma_relative_error = (
        abs(gamma_measured / mapping.physical.gamma - 1.0) if np.isfinite(gamma_measured) else np.nan
    )

    attenuation_logabs = decay_fit["decay_rate"]
    attenuation_prony = _prony_decay_rate(time_array, pressure_array, settings.fit_start, settings.fit_stop)
    # Prony is the window-free caliber (== exact one-step modal eigenvalue); the
    # log|p'| slope is kept as a secondary field and used only as a fallback when
    # Prony is unavailable (too few samples / degenerate fit).
    attenuation_caliber = "prony" if np.isfinite(attenuation_prony) else "logabs"
    attenuation_measured = attenuation_prony if np.isfinite(attenuation_prony) else attenuation_logabs
    attenuation_target = _reference_acoustic_attenuation_rate(solver, k_mag)
    attenuation_relative_error = (
        abs(attenuation_measured / attenuation_target - 1.0)
        if np.isfinite(attenuation_measured) and attenuation_target != 0.0
        else np.nan
    )
    passed = (
        first_invalid_step is None
        and not nan_detected
        and np.isfinite(sound_speed_relative_error)
        and sound_speed_relative_error <= settings.sound_speed_tolerance
        and np.isfinite(gamma_relative_error)
        and gamma_relative_error <= settings.gamma_tolerance
        and np.isfinite(attenuation_measured)
    )

    return {
        "direction": direction,
        "status": "PASSED" if passed else "FAILED",
        "sound_speed_target_lu": c_target,
        "sound_speed_measured_lu": float(sound_speed_measured) if np.isfinite(sound_speed_measured) else np.nan,
        "sound_speed_relative_error": (
            float(sound_speed_relative_error) if np.isfinite(sound_speed_relative_error) else np.nan
        ),
        "gamma_target": float(mapping.physical.gamma),
        "gamma_measured": float(gamma_measured) if np.isfinite(gamma_measured) else np.nan,
        "gamma_relative_error": float(gamma_relative_error) if np.isfinite(gamma_relative_error) else np.nan,
        "acoustic_attenuation_measured_lu": (
            float(attenuation_measured) if np.isfinite(attenuation_measured) else np.nan
        ),
        "acoustic_attenuation_caliber": attenuation_caliber,
        "acoustic_attenuation_logabs_lu": (
            float(attenuation_logabs) if np.isfinite(attenuation_logabs) else np.nan
        ),
        "acoustic_attenuation_reference_lu": float(attenuation_target),
        "acoustic_attenuation_relative_error": (
            float(attenuation_relative_error) if np.isfinite(attenuation_relative_error) else np.nan
        ),
        "acoustic_attenuation_target_policy": ACOUSTIC_ATTENUATION_TARGET_POLICY,
        "acoustic_attenuation_target_coeff_lu": _matched_nsf_acoustic_attenuation_coeff_lu(mapping),
        "attenuation_status": ACOUSTIC_ATTENUATION_STATUS,
        "mode_index": settings.mode_index,
        "background_velocity_lu": list(settings.background_velocity_lu),
        "background_theta_lu": float(theta0),
        "background_rho_lu": float(rho0),
        "background_path": settings.background_path,
        "background_advection_lu": background_advection,
        "lab_phase_speed_lu": float(lab_phase_speed) if np.isfinite(lab_phase_speed) else np.nan,
        "intrinsic_phase_speed_lu": (
            float(intrinsic_phase_speed) if np.isfinite(intrinsic_phase_speed) else np.nan
        ),
        "k_mag_lu": k_mag,
        "fitting_window": phase_fit["fitting_window"],
        "phase_residual_norm": phase_fit["phase_residual_norm"],
        "decay_residual_norm": decay_fit["residual_norm"],
        "fit_sample_count": min(phase_fit["fit_sample_count"], decay_fit["fit_sample_count"]),
        "sample_count": len(times),
        "pressure_amplitude_initial": float(abs(pressure_array[0])) if pressure_array.size else np.nan,
        "pressure_amplitude_final": float(abs(pressure_array[-1])) if pressure_array.size else np.nan,
        "density_amplitude_initial": (
            float(abs(np.asarray(density_amplitudes, dtype=complex)[0])) if density_amplitudes else np.nan
        ),
        "theta_amplitude_initial": (
            float(abs(np.asarray(theta_amplitudes, dtype=complex)[0])) if theta_amplitudes else np.nan
        ),
        "velocity_amplitude_initial": (
            float(abs(np.asarray(velocity_amplitudes, dtype=complex)[0])) if velocity_amplitudes else np.nan
        ),
        "first_invalid_step": first_invalid_step,
        "nan_detected": nan_detected,
        "negative_theta_detected": negative_theta_detected,
        "min_theta_lu": float(min_theta),
        "max_theta_lu": float(max_theta),
        "max_mach": float(max_mach),
        "max_mean_pressure_drift": float(max_mean_pressure_drift),
        "clipping_used": False,
    }


def measure_acoustic_wave(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings_from_config(config)
    direction_results = {
        direction: measure_acoustic_wave_direction(config, direction, settings)
        for direction in settings.directions
    }
    finite_speeds = [
        result["sound_speed_measured_lu"]
        for result in direction_results.values()
        if np.isfinite(result["sound_speed_measured_lu"])
    ]
    c_target = next(iter(direction_results.values()))["sound_speed_target_lu"]
    direction_difference = (
        float((max(finite_speeds) - min(finite_speeds)) / c_target) if len(finite_speeds) >= 2 else np.nan
    )
    direction_pass = (not np.isfinite(direction_difference)) or direction_difference <= settings.direction_tolerance
    all_direction_pass = all(result["status"] == "PASSED" for result in direction_results.values())
    passed = all_direction_pass and direction_pass

    baseline = direction_results.get("x", next(iter(direction_results.values())))
    max_speed_error = max(
        (
            result["sound_speed_relative_error"]
            for result in direction_results.values()
            if np.isfinite(result["sound_speed_relative_error"])
        ),
        default=np.nan,
    )
    max_gamma_error = max(
        (
            result["gamma_relative_error"]
            for result in direction_results.values()
            if np.isfinite(result["gamma_relative_error"])
        ),
        default=np.nan,
    )
    attenuation_errors = [
        result["acoustic_attenuation_relative_error"]
        for result in direction_results.values()
        if np.isfinite(result["acoustic_attenuation_relative_error"])
    ]
    first_invalid_steps = [
        result["first_invalid_step"]
        for result in direction_results.values()
        if result["first_invalid_step"] is not None
    ]
    fitting_windows = {name: result["fitting_window"] for name, result in direction_results.items()}
    phase_residuals = [
        result["phase_residual_norm"]
        for result in direction_results.values()
        if np.isfinite(result["phase_residual_norm"])
    ]
    decay_residuals = [
        result["decay_residual_norm"]
        for result in direction_results.values()
        if np.isfinite(result["decay_residual_norm"])
    ]

    return {
        "p2_06_status": "PASSED" if passed else "FAILED",
        "sound_speed_target_lu": float(c_target),
        "sound_speed_measured_lu": baseline["sound_speed_measured_lu"],
        "sound_speed_relative_error": float(max_speed_error) if np.isfinite(max_speed_error) else np.nan,
        "gamma_target": baseline["gamma_target"],
        "gamma_measured": baseline["gamma_measured"],
        "gamma_relative_error": float(max_gamma_error) if np.isfinite(max_gamma_error) else np.nan,
        "acoustic_attenuation_measured_lu": baseline["acoustic_attenuation_measured_lu"],
        "acoustic_attenuation_caliber": baseline.get("acoustic_attenuation_caliber", "prony"),
        "acoustic_attenuation_logabs_lu": baseline.get("acoustic_attenuation_logabs_lu", np.nan),
        "acoustic_attenuation_reference_lu": baseline["acoustic_attenuation_reference_lu"],
        "acoustic_attenuation_relative_error": (
            float(max(attenuation_errors)) if attenuation_errors else np.nan
        ),
        "acoustic_attenuation_target_policy": baseline["acoustic_attenuation_target_policy"],
        "acoustic_attenuation_target_coeff_lu": baseline["acoustic_attenuation_target_coeff_lu"],
        "attenuation_status": ACOUSTIC_ATTENUATION_STATUS,
        "baseline_direction": baseline["direction"],
        "baseline_sound_speed_relative_error": baseline["sound_speed_relative_error"],
        "baseline_gamma_relative_error": baseline["gamma_relative_error"],
        "mode_index": settings.mode_index,
        "background_velocity_lu": list(settings.background_velocity_lu),
        "fitting_window": fitting_windows,
        "phase_residual_norm": float(max(phase_residuals)) if phase_residuals else np.nan,
        "decay_residual_norm": float(max(decay_residuals)) if decay_residuals else np.nan,
        "directions": list(settings.directions),
        "direction_difference": direction_difference,
        "direction_results": direction_results,
        "first_invalid_step": min(first_invalid_steps) if first_invalid_steps else None,
        "nan_detected": any(result["nan_detected"] for result in direction_results.values()),
        "negative_theta_detected": any(result["negative_theta_detected"] for result in direction_results.values()),
        "clipping_used": any(result["clipping_used"] for result in direction_results.values()),
        "acceptance": {
            "sound_speed_relative_error_max": settings.sound_speed_tolerance,
            "gamma_relative_error_max": settings.gamma_tolerance,
            "direction_difference_max": settings.direction_tolerance,
            "attenuation_is_diagnostic_only": True,
            "requires_no_nan": True,
            "requires_no_clipping": True,
            "requires_positive_theta": True,
        },
    }
