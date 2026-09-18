"""Real periodic shear-wave viscosity measurement for Phase 2 P2-4."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.solver import GasSolver2D


DEFAULT_DIRECTIONS = ("x", "y", "diagonal")
DEFAULT_RELATIVE_TOLERANCE = 0.05
DEFAULT_DIRECTION_TOLERANCE = 0.05


@dataclass(frozen=True)
class ShearWaveSettings:
    nx: int = 64
    ny: int = 64
    steps: int = 120
    sample_interval: int = 1
    mode_index: int = 1
    amplitude: float = 1.0e-5
    fit_start: int = 10
    fit_stop: int | None = None
    directions: tuple[str, ...] = DEFAULT_DIRECTIONS
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE
    background_velocity_lu: tuple[float, float] = (0.0, 0.0)
    # Phase_5 G0 (contract §5.1): uniform non-reference background on the SAME
    # frozen mapping. None -> theta_ref (behaviour identical to Phase_2).
    background_theta_lu: float | None = None
    background_path: str = "isobaric"  # "isobaric" | "equal_density"
    # G0-QM (2026-09-08): opt-in amplitude series (default False -> result dict unchanged).
    return_series: bool = False


def _settings_from_config(config: dict[str, Any]) -> ShearWaveSettings:
    p2 = dict(config.get("p2_04_shear_wave", {}) or {})
    initial = dict(config.get("initial_condition", {}) or {})
    if initial.get("type") == "shear_wave":
        p2.setdefault("amplitude", initial.get("amplitude"))
        p2.setdefault("mode_index", initial.get("wavenumber_mode"))
        if "directions" not in p2 and "direction" in initial:
            p2["directions"] = [initial["direction"]]

    directions = p2.get("directions", DEFAULT_DIRECTIONS)
    if isinstance(directions, str):
        directions = [directions]
    directions_tuple = tuple(str(item) for item in directions)
    if not directions_tuple:
        directions_tuple = DEFAULT_DIRECTIONS

    return ShearWaveSettings(
        nx=int(p2.get("nx", 64)),
        ny=int(p2.get("ny", 64)),
        steps=int(p2.get("steps", 120)),
        sample_interval=int(p2.get("sample_interval", 1)),
        mode_index=int(p2.get("mode_index", p2.get("wavenumber_mode", 1))),
        amplitude=float(p2.get("amplitude", 1.0e-5)),
        fit_start=int(p2.get("fit_start", 10)),
        fit_stop=None if p2.get("fit_stop", None) is None else int(p2["fit_stop"]),
        directions=directions_tuple,
        relative_tolerance=float(p2.get("relative_tolerance", DEFAULT_RELATIVE_TOLERANCE)),
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


def _simulation_config(config: dict[str, Any], settings: ShearWaveSettings, direction: str) -> dict[str, Any]:
    sim_config = deepcopy(config)
    sim_config["numerics"] = dict(sim_config.get("numerics", {}) or {})
    sim_config["numerics"]["nx"] = settings.nx
    sim_config["numerics"]["ny"] = settings.ny
    sim_config["case"] = dict(sim_config.get("case", {}) or {})
    sim_config["case"]["name"] = f"p2_04_shear_wave_{direction}"
    return sim_config


def _mesh(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    return np.meshgrid(x, y)


def _direction_phase_and_transverse(
    direction: str,
    ny: int,
    nx: int,
    mode_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_grid, y_grid = _mesh(ny, nx)
    if direction == "x":
        kx = 2.0 * np.pi * mode_index / nx
        phase = kx * x_grid
        transverse = np.array([0.0, 1.0])
        k2 = kx * kx
    elif direction == "y":
        ky = 2.0 * np.pi * mode_index / ny
        phase = ky * y_grid
        transverse = np.array([1.0, 0.0])
        k2 = ky * ky
    elif direction == "diagonal":
        kx = 2.0 * np.pi * mode_index / nx
        ky = 2.0 * np.pi * mode_index / ny
        phase = kx * x_grid + ky * y_grid
        transverse = np.array([1.0, -1.0]) / np.sqrt(2.0)
        k2 = kx * kx + ky * ky
    else:
        raise ValueError("direction must be x, y, or diagonal")
    return phase, transverse, float(k2)


def _initialize_shear_wave(solver: GasSolver2D, settings: ShearWaveSettings, direction: str) -> float:
    phase, transverse, k2 = _direction_phase_and_transverse(
        direction,
        solver.ny,
        solver.nx,
        settings.mode_index,
    )
    theta_b, rho_b = _background_state(solver.mapping, settings)
    rho = np.full((solver.ny, solver.nx), rho_b, dtype=float)
    theta = np.full((solver.ny, solver.nx), theta_b, dtype=float)
    u = np.broadcast_to(np.asarray(settings.background_velocity_lu, dtype=float), (solver.ny, solver.nx, 2)).copy()
    u += settings.amplitude * np.sin(phase)[..., None] * transverse[None, None, :]
    solver.initialize_from_macro(rho, u, theta)
    return k2


def _modal_amplitude_from_state(u: np.ndarray, direction: str, mode_index: int) -> complex:
    ny, nx = u.shape[:2]
    phase, transverse, _ = _direction_phase_and_transverse(direction, ny, nx, mode_index)
    transverse_velocity = np.einsum("...i,i->...", u, transverse)
    transverse_velocity = transverse_velocity - np.mean(transverse_velocity)
    return complex((2.0 / transverse_velocity.size) * np.sum(transverse_velocity * np.exp(-1j * phase)))


def _fit_decay(times: np.ndarray, amplitudes: np.ndarray, start: int, stop: int | None) -> dict[str, Any]:
    finite = np.isfinite(times) & np.isfinite(np.abs(amplitudes))
    if stop is None:
        stop_value = int(np.max(times[finite])) if np.any(finite) else start
    else:
        stop_value = stop
    mask = finite & (times >= start) & (times <= stop_value)
    if np.count_nonzero(mask) < 3:
        return {
            "decay_rate": np.nan,
            "intercept": np.nan,
            "residual_norm": np.nan,
            "fitting_window": [int(start), int(stop_value)],
            "fit_sample_count": int(np.count_nonzero(mask)),
        }

    t_fit = times[mask]
    y_fit = np.log(np.maximum(np.abs(amplitudes[mask]), 1.0e-300))
    matrix = np.column_stack((np.ones_like(t_fit), t_fit))
    coeffs, *_ = np.linalg.lstsq(matrix, y_fit, rcond=None)
    intercept, slope = coeffs
    decay_rate = float(-slope)
    residual = y_fit - (intercept + slope * t_fit)
    return {
        "decay_rate": decay_rate,
        "intercept": float(intercept),
        "residual_norm": float(np.sqrt(np.mean(residual * residual))),
        "fitting_window": [int(np.min(t_fit)), int(np.max(t_fit))],
        "fit_sample_count": int(t_fit.size),
    }


def measure_shear_wave_direction(config: dict[str, Any], direction: str, settings: ShearWaveSettings) -> dict[str, Any]:
    if settings.amplitude > 1.0e-4:
        raise ValueError("P2-4 shear-wave amplitude must satisfy A_u/c_lu <= 1e-4")
    if settings.sample_interval <= 0:
        raise ValueError("sample_interval must be positive")

    solver = GasSolver2D(_simulation_config(config, settings, direction))
    k2 = _initialize_shear_wave(solver, settings, direction)
    theta_bg, rho_bg = _background_state(solver.mapping, settings)
    times: list[int] = []
    amplitudes: list[complex] = []
    first_invalid_step: int | None = None
    nan_detected = False
    negative_theta_detected = False
    min_theta = np.inf
    max_theta = -np.inf
    max_rho_drift = 0.0

    for step in range(settings.steps + 1):
        with np.errstate(all="ignore"):
            macro = solver.get_macro()
        finite = bool(np.isfinite(macro.rho).all() and np.isfinite(macro.u).all() and np.isfinite(macro.theta).all())
        nan_detected = nan_detected or not finite
        theta_min = float(np.nanmin(macro.theta)) if macro.theta.size else np.nan
        theta_max = float(np.nanmax(macro.theta)) if macro.theta.size else np.nan
        min_theta = min(min_theta, theta_min)
        max_theta = max(max_theta, theta_max)
        max_rho_drift = max(max_rho_drift, float(np.nanmax(np.abs(macro.rho - rho_bg))))
        if (not finite) or theta_min <= 0.0:
            first_invalid_step = step
            negative_theta_detected = negative_theta_detected or bool(theta_min <= 0.0)
            break

        if step % settings.sample_interval == 0:
            times.append(step)
            amplitudes.append(_modal_amplitude_from_state(macro.u, direction, settings.mode_index))
        if step < settings.steps:
            with np.errstate(all="ignore"):
                solver.step()

    time_array = np.asarray(times, dtype=float)
    amplitude_array = np.asarray(amplitudes, dtype=complex)
    fit = _fit_decay(time_array, amplitude_array, settings.fit_start, settings.fit_stop)
    nu_measured = fit["decay_rate"] / k2 if np.isfinite(fit["decay_rate"]) and k2 > 0.0 else np.nan
    nu_target = solver.mapping.nu_lu
    relative_error = abs(nu_measured / nu_target - 1.0) if np.isfinite(nu_measured) else np.nan
    passed = (
        first_invalid_step is None
        and not nan_detected
        and np.isfinite(relative_error)
        and relative_error <= settings.relative_tolerance
    )

    result = {
        "direction": direction,
        "status": "PASSED" if passed else "FAILED",
        "nu_target_lu": float(nu_target),
        "nu_measured_lu": float(nu_measured) if np.isfinite(nu_measured) else np.nan,
        "relative_error": float(relative_error) if np.isfinite(relative_error) else np.nan,
        "background_theta_lu": float(theta_bg),
        "background_rho_lu": float(rho_bg),
        "background_path": settings.background_path,
        "mode_index": settings.mode_index,
        "background_velocity_lu": list(settings.background_velocity_lu),
        "k2_lu": k2,
        "fitting_window": fit["fitting_window"],
        "residual_norm": fit["residual_norm"],
        "fit_sample_count": fit["fit_sample_count"],
        "sample_count": len(times),
        "first_invalid_step": first_invalid_step,
        "nan_detected": nan_detected,
        "negative_theta_detected": negative_theta_detected,
        "min_theta_lu": float(min_theta),
        "max_theta_lu": float(max_theta),
        "max_rho_drift": float(max_rho_drift),
        "clipping_used": False,
    }
    if settings.return_series:
        result["series"] = {
            "times": [int(t) for t in time_array],
            "abs_amplitude": [float(v) for v in np.abs(amplitude_array)],
        }
    return result


def measure_shear_wave(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings_from_config(config)
    direction_results = {
        direction: measure_shear_wave_direction(config, direction, settings) for direction in settings.directions
    }
    finite_nu = [
        result["nu_measured_lu"]
        for result in direction_results.values()
        if np.isfinite(result["nu_measured_lu"])
    ]
    nu_target = next(iter(direction_results.values()))["nu_target_lu"]
    direction_difference = (
        float((max(finite_nu) - min(finite_nu)) / nu_target) if len(finite_nu) >= 2 else np.nan
    )
    direction_pass = (not np.isfinite(direction_difference)) or direction_difference <= settings.direction_tolerance
    all_direction_pass = all(result["status"] == "PASSED" for result in direction_results.values())
    passed = all_direction_pass and direction_pass

    baseline = direction_results.get("x", next(iter(direction_results.values())))
    max_relative_error = max(
        (
            result["relative_error"]
            for result in direction_results.values()
            if np.isfinite(result["relative_error"])
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
        "p2_04_status": "PASSED" if passed else "FAILED",
        "nu_target_lu": float(nu_target),
        "nu_measured_lu": baseline["nu_measured_lu"],
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
        "first_invalid_step": min(first_invalid_steps) if first_invalid_steps else None,
        "nan_detected": any(result["nan_detected"] for result in direction_results.values()),
        "negative_theta_detected": any(result["negative_theta_detected"] for result in direction_results.values()),
        "clipping_used": any(result["clipping_used"] for result in direction_results.values()),
        "acceptance": {
            "relative_error_max": settings.relative_tolerance,
            "direction_difference_max": settings.direction_tolerance,
            "requires_no_nan": True,
            "requires_no_clipping": True,
            "requires_positive_theta": True,
        },
    }
