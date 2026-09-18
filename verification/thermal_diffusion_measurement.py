"""Real isobaric thermal diffusion and Fourier-law checks for Phase 2 P2-5."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.solver import GasSolver2D
from verification.shear_wave_measurement import _fit_decay


DEFAULT_DIRECTIONS = ("x", "y")
DEFAULT_RELATIVE_TOLERANCE = 0.05
DEFAULT_HEAT_FLUX_TOLERANCE = 0.05


@dataclass(frozen=True)
class ThermalDiffusionSettings:
    nx: int = 64
    ny: int = 64
    steps: int = 320
    sample_interval: int = 1
    mode_index: int = 1
    amplitude: float = 1.0e-5
    fit_start: int = 10
    fit_stop: int | None = None
    directions: tuple[str, ...] = DEFAULT_DIRECTIONS
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE
    heat_flux_tolerance: float = DEFAULT_HEAT_FLUX_TOLERANCE
    background_velocity_lu: tuple[float, float] = (0.0, 0.0)
    # Phase_5 G0 (contract §5.1): uniform non-reference background on the SAME
    # frozen mapping. None -> theta_ref (behaviour identical to Phase_2).
    background_theta_lu: float | None = None
    background_path: str = "isobaric"  # "isobaric": rho_b=rho_ref*theta_ref/theta_b | "equal_density"
    # G0-QM (2026-09-08): opt-in return of the sampled modal amplitude series for
    # readout diagnostics (local-slope variation).  Default False -> result dict unchanged.
    return_series: bool = False


def _settings_from_config(config: dict[str, Any]) -> ThermalDiffusionSettings:
    p2 = dict(config.get("p2_05_thermal_diffusion", {}) or {})
    initial = dict(config.get("initial_condition", {}) or {})
    if initial.get("type") == "isobaric_thermal_sine":
        p2.setdefault("amplitude", initial.get("amplitude"))
        p2.setdefault("mode_index", initial.get("wavenumber_mode"))
        if "directions" not in p2 and "direction" in initial:
            p2["directions"] = [initial["direction"]]

    directions = p2.get("directions", DEFAULT_DIRECTIONS)
    if isinstance(directions, str):
        directions = [directions]
    directions_tuple = tuple(str(item) for item in directions) or DEFAULT_DIRECTIONS

    return ThermalDiffusionSettings(
        nx=int(p2.get("nx", 64)),
        ny=int(p2.get("ny", 64)),
        steps=int(p2.get("steps", 320)),
        sample_interval=int(p2.get("sample_interval", 1)),
        mode_index=int(p2.get("mode_index", p2.get("wavenumber_mode", 1))),
        amplitude=float(p2.get("amplitude", 1.0e-5)),
        fit_start=int(p2.get("fit_start", 10)),
        fit_stop=None if p2.get("fit_stop", None) is None else int(p2["fit_stop"]),
        directions=directions_tuple,
        relative_tolerance=float(p2.get("relative_tolerance", DEFAULT_RELATIVE_TOLERANCE)),
        heat_flux_tolerance=float(p2.get("heat_flux_tolerance", DEFAULT_HEAT_FLUX_TOLERANCE)),
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


def _simulation_config(config: dict[str, Any], settings: ThermalDiffusionSettings, direction: str) -> dict[str, Any]:
    sim_config = deepcopy(config)
    sim_config["numerics"] = dict(sim_config.get("numerics", {}) or {})
    sim_config["numerics"]["nx"] = settings.nx
    sim_config["numerics"]["ny"] = settings.ny
    sim_config["case"] = dict(sim_config.get("case", {}) or {})
    sim_config["case"]["name"] = f"p2_05_thermal_diffusion_{direction}"
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
        phase = kx * x_grid + ky * y_grid
        k_mag = float(np.sqrt(kx * kx + ky * ky))
        unit = np.array([kx, ky], dtype=float) / k_mag
    else:
        raise ValueError("direction must be x, y, or diagonal")
    return phase, unit, float(k_mag)


def _modal_amplitude_2d(field: np.ndarray, direction: str, mode_index: int) -> complex:
    arr = np.asarray(field, dtype=float)
    phase, _, _ = _direction_phase_and_unit(direction, arr.shape[0], arr.shape[1], mode_index)
    centered = arr - np.mean(arr)
    return complex((2.0 / centered.size) * np.sum(centered * np.exp(-1j * phase)))


def _initialize_isobaric_thermal_wave(
    solver: GasSolver2D,
    settings: ThermalDiffusionSettings,
    direction: str,
) -> float:
    phase, _, k_mag = _direction_phase_and_unit(direction, solver.ny, solver.nx, settings.mode_index)
    theta_b, rho_b = _background_state(solver.mapping, settings)
    theta = theta_b * (1.0 + settings.amplitude * np.sin(phase))
    p_b = rho_b * theta_b
    rho = p_b / theta  # locally isobaric perturbation around the uniform background
    u = np.broadcast_to(np.asarray(settings.background_velocity_lu, dtype=float), (solver.ny, solver.nx, 2)).copy()
    solver.initialize_from_macro(rho, u, theta)
    return k_mag


def _fourier_heat_flux_coefficient_lu(solver: GasSolver2D) -> float:
    mapping = solver.mapping
    return (
        mapping.physical.kg_W_mK
        * mapping.temperature_scale
        / (mapping.lattice.dx_m * mapping.heat_flux_scale)
    )


def _heat_flux_ratio(
    theta_amplitudes: np.ndarray,
    heat_flux_amplitudes: np.ndarray,
    times: np.ndarray,
    *,
    k_mag: float,
    fourier_coeff_lu: float,
    start: int,
    stop: int | None,
) -> dict[str, Any]:
    stop_value = int(np.max(times)) if stop is None and times.size else start
    if stop is not None:
        stop_value = stop
    mask = (times >= start) & (times <= stop_value)
    expected = -1j * k_mag * fourier_coeff_lu * theta_amplitudes[mask]
    valid = np.abs(expected) > 1.0e-300
    if np.count_nonzero(valid) == 0:
        return {
            "heat_flux_ratio": np.nan,
            "heat_flux_relative_error": np.nan,
            "heat_flux_sign_passed": False,
        }
    ratios = heat_flux_amplitudes[mask][valid] / expected[valid]
    mean_ratio = complex(np.mean(ratios))
    return {
        "heat_flux_ratio": mean_ratio,
        "heat_flux_relative_error": float(abs(mean_ratio - 1.0)),
        "heat_flux_sign_passed": bool(np.real(mean_ratio) > 0.0),
    }


def measure_thermal_diffusion_direction(
    config: dict[str, Any],
    direction: str,
    settings: ThermalDiffusionSettings,
) -> dict[str, Any]:
    if settings.amplitude > 1.0e-4:
        raise ValueError("P2-5 thermal amplitude must satisfy A_T/theta_ref <= 1e-4")
    if settings.sample_interval <= 0:
        raise ValueError("sample_interval must be positive")

    solver = GasSolver2D(_simulation_config(config, settings, direction))
    k_mag = _initialize_isobaric_thermal_wave(solver, settings, direction)
    times: list[int] = []
    theta_amplitudes: list[complex] = []
    heat_flux_amplitudes: list[complex] = []
    first_invalid_step: int | None = None
    nan_detected = False
    negative_theta_detected = False
    min_theta = np.inf
    max_theta = -np.inf
    max_pressure_drift = 0.0

    theta0, rho_bg = _background_state(solver.mapping, settings)
    p0 = rho_bg * theta0
    _, unit, _ = _direction_phase_and_unit(direction, solver.ny, solver.nx, settings.mode_index)

    for step in range(settings.steps + 1):
        with np.errstate(all="ignore"):
            macro = solver.get_macro()
        finite = bool(np.isfinite(macro.rho).all() and np.isfinite(macro.u).all() and np.isfinite(macro.theta).all())
        nan_detected = nan_detected or not finite
        theta_min = float(np.nanmin(macro.theta)) if macro.theta.size else np.nan
        theta_max = float(np.nanmax(macro.theta)) if macro.theta.size else np.nan
        min_theta = min(min_theta, theta_min)
        max_theta = max(max_theta, theta_max)
        max_pressure_drift = max(max_pressure_drift, float(np.nanmax(np.abs(macro.p - p0))))
        if (not finite) or theta_min <= 0.0:
            first_invalid_step = step
            negative_theta_detected = negative_theta_detected or bool(theta_min <= 0.0)
            break

        if step % settings.sample_interval == 0:
            q = solver.get_heat_flux_lu()
            q_normal = np.einsum("...i,i->...", q, unit)
            times.append(step)
            theta_amplitudes.append(_modal_amplitude_2d(macro.theta - theta0, direction, settings.mode_index))
            heat_flux_amplitudes.append(_modal_amplitude_2d(q_normal, direction, settings.mode_index))
        if step < settings.steps:
            with np.errstate(all="ignore"):
                solver.step()

    time_array = np.asarray(times, dtype=float)
    theta_array = np.asarray(theta_amplitudes, dtype=complex)
    heat_flux_array = np.asarray(heat_flux_amplitudes, dtype=complex)
    fit = _fit_decay(time_array, theta_array, settings.fit_start, settings.fit_stop)
    alpha_measured = fit["decay_rate"] / (k_mag * k_mag) if np.isfinite(fit["decay_rate"]) else np.nan
    alpha_target = solver.mapping.alpha_lu
    relative_error = abs(alpha_measured / alpha_target - 1.0) if np.isfinite(alpha_measured) else np.nan
    heat_flux = _heat_flux_ratio(
        theta_array,
        heat_flux_array,
        time_array,
        k_mag=k_mag,
        fourier_coeff_lu=_fourier_heat_flux_coefficient_lu(solver),
        start=settings.fit_start,
        stop=settings.fit_stop,
    )
    heat_flux_passed = (
        heat_flux["heat_flux_sign_passed"]
        and np.isfinite(heat_flux["heat_flux_relative_error"])
        and heat_flux["heat_flux_relative_error"] <= settings.heat_flux_tolerance
    )
    passed = (
        first_invalid_step is None
        and not nan_detected
        and np.isfinite(relative_error)
        and relative_error <= settings.relative_tolerance
        and heat_flux_passed
    )

    heat_flux_ratio = heat_flux["heat_flux_ratio"]
    result = {
        "direction": direction,
        "status": "PASSED" if passed else "FAILED",
        "alpha_target_lu": float(alpha_target),
        "alpha_measured_lu": float(alpha_measured) if np.isfinite(alpha_measured) else np.nan,
        "relative_error": float(relative_error) if np.isfinite(relative_error) else np.nan,
        "mode_index": settings.mode_index,
        "background_velocity_lu": list(settings.background_velocity_lu),
        "background_theta_lu": float(theta0),
        "background_rho_lu": float(rho_bg),
        "background_path": settings.background_path,
        "k_mag_lu": k_mag,
        "fitting_window": fit["fitting_window"],
        "residual_norm": fit["residual_norm"],
        "fit_sample_count": fit["fit_sample_count"],
        "sample_count": len(times),
        "heat_flux_ratio_real": float(np.real(heat_flux_ratio)) if np.isfinite(heat_flux_ratio) else np.nan,
        "heat_flux_ratio_imag": float(np.imag(heat_flux_ratio)) if np.isfinite(heat_flux_ratio) else np.nan,
        "heat_flux_relative_error": heat_flux["heat_flux_relative_error"],
        "heat_flux_sign_passed": heat_flux["heat_flux_sign_passed"],
        "first_invalid_step": first_invalid_step,
        "nan_detected": nan_detected,
        "negative_theta_detected": negative_theta_detected,
        "min_theta_lu": float(min_theta),
        "max_theta_lu": float(max_theta),
        "max_pressure_drift": float(max_pressure_drift),
        "clipping_used": False,
    }
    if settings.return_series:
        result["series"] = {
            "times": [int(t) for t in time_array],
            "abs_amplitude": [float(v) for v in np.abs(theta_array)],
        }
    return result


def measure_thermal_diffusion(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings_from_config(config)
    direction_results = {
        direction: measure_thermal_diffusion_direction(config, direction, settings)
        for direction in settings.directions
    }
    finite_alpha = [
        result["alpha_measured_lu"]
        for result in direction_results.values()
        if np.isfinite(result["alpha_measured_lu"])
    ]
    alpha_target = next(iter(direction_results.values()))["alpha_target_lu"]
    direction_difference = (
        float((max(finite_alpha) - min(finite_alpha)) / alpha_target) if len(finite_alpha) >= 2 else np.nan
    )
    all_direction_pass = all(result["status"] == "PASSED" for result in direction_results.values())

    baseline = direction_results.get("x", next(iter(direction_results.values())))
    max_relative_error = max(
        (
            result["relative_error"]
            for result in direction_results.values()
            if np.isfinite(result["relative_error"])
        ),
        default=np.nan,
    )
    max_heat_flux_error = max(
        (
            result["heat_flux_relative_error"]
            for result in direction_results.values()
            if np.isfinite(result["heat_flux_relative_error"])
        ),
        default=np.nan,
    )
    first_invalid_steps = [
        result["first_invalid_step"]
        for result in direction_results.values()
        if result["first_invalid_step"] is not None
    ]
    fitting_windows = {name: result["fitting_window"] for name, result in direction_results.items()}
    residuals = [
        result["residual_norm"]
        for result in direction_results.values()
        if np.isfinite(result["residual_norm"])
    ]

    return {
        "p2_05_status": "PASSED" if all_direction_pass else "FAILED",
        "alpha_target_lu": float(alpha_target),
        "alpha_measured_lu": baseline["alpha_measured_lu"],
        "relative_error": float(max_relative_error) if np.isfinite(max_relative_error) else np.nan,
        "baseline_direction": baseline["direction"],
        "baseline_relative_error": baseline["relative_error"],
        "mode_index": settings.mode_index,
        "background_velocity_lu": list(settings.background_velocity_lu),
        "fitting_window": fitting_windows,
        "residual_norm": float(max(residuals)) if residuals else np.nan,
        "directions": list(settings.directions),
        "direction_difference": direction_difference,
        "direction_results": direction_results,
        "heat_flux_relative_error": float(max_heat_flux_error) if np.isfinite(max_heat_flux_error) else np.nan,
        "heat_flux_sign_passed": all(result["heat_flux_sign_passed"] for result in direction_results.values()),
        "first_invalid_step": min(first_invalid_steps) if first_invalid_steps else None,
        "nan_detected": any(result["nan_detected"] for result in direction_results.values()),
        "negative_theta_detected": any(result["negative_theta_detected"] for result in direction_results.values()),
        "clipping_used": any(result["clipping_used"] for result in direction_results.values()),
        "acceptance": {
            "relative_error_max": settings.relative_tolerance,
            "heat_flux_relative_error_max": settings.heat_flux_tolerance,
            "requires_no_nan": True,
            "requires_no_clipping": True,
            "requires_positive_theta": True,
            "requires_fourier_sign": True,
        },
    }
