
"""Phase_3 film-gas coupling utilities."""

from .drive import ConstantDrive, GaussianPulseDrive, SinusoidalDrive, StepDrive
from .energy_audit import EnergyAuditResult, audit_film_energy
from .film_ode import (
    FilmOdeParams,
    FilmTrajectory,
    adiabatic_ramp_solution,
    energy_residual_cumulative,
    euler_step,
    film_rhs,
    heun_step,
    integrate_film_ode,
    integrated_energy_residual,
    linear_leak_step_solution,
    ode_pointwise_residual,
    rk4_step,
    sinusoidal_steady_temperature,
    sinusoidal_temperature_hat,
)
from .conjugate import LevelCCouplingResult, run_levelc_predictor_corrector

__all__ = [
    "ConstantDrive",
    "EnergyAuditResult",
    "FilmOdeParams",
    "FilmTrajectory",
    "GaussianPulseDrive",
    "LevelCCouplingResult",
    "SinusoidalDrive",
    "StepDrive",
    "adiabatic_ramp_solution",
    "audit_film_energy",
    "energy_residual_cumulative",
    "euler_step",
    "film_rhs",
    "heun_step",
    "integrate_film_ode",
    "integrated_energy_residual",
    "linear_leak_step_solution",
    "ode_pointwise_residual",
    "rk4_step",
    "run_levelc_predictor_corrector",
    "sinusoidal_steady_temperature",
    "sinusoidal_temperature_hat",
]
