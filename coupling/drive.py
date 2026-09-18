"""Phase_3 film power-density drive signals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


class DriveSignal(Protocol):
    """Callable power-density signal in SI units, W/m^2."""

    def __call__(self, t_si: float) -> float:
        ...


@dataclass(frozen=True)
class ConstantDrive:
    """Constant film input power density."""

    power_density_si: float

    def __call__(self, t_si: float) -> float:
        return float(self.power_density_si)


@dataclass(frozen=True)
class StepDrive:
    """Step drive ``baseline + power_density * H(t - t0)``."""

    power_density_si: float
    t0_si: float = 0.0
    baseline_si: float = 0.0

    def __call__(self, t_si: float) -> float:
        if float(t_si) < self.t0_si:
            return float(self.baseline_si)
        return float(self.baseline_si + self.power_density_si)


@dataclass(frozen=True)
class SinusoidalDrive:
    """Harmonic drive using ``P(t)=mean+Re[P_hat exp(i Omega t)]``."""

    mean_si: float
    amplitude_hat_si: complex
    frequency_hz: float

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")

    @property
    def omega_si(self) -> float:
        return 2.0 * math.pi * float(self.frequency_hz)

    def __call__(self, t_si: float) -> float:
        phase = complex(math.cos(self.omega_si * t_si), math.sin(self.omega_si * t_si))
        return float(self.mean_si + (self.amplitude_hat_si * phase).real)


@dataclass(frozen=True)
class GaussianPulseDrive:
    """Gaussian pulse ``baseline + amplitude * exp(-((t-t0)/sigma)^2)``."""

    amplitude_si: float
    t0_si: float
    sigma_si: float
    baseline_si: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_si <= 0.0:
            raise ValueError("sigma_si must be positive")

    def __call__(self, t_si: float) -> float:
        arg = (float(t_si) - self.t0_si) / self.sigma_si
        return float(self.baseline_si + self.amplitude_si * math.exp(-(arg * arg)))


def evaluate_drive(drive: DriveSignal | float, t_si: float) -> float:
    """Evaluate either a callable drive or a scalar power density."""

    if callable(drive):
        return float(drive(float(t_si)))
    return float(drive)


__all__ = [
    "ConstantDrive",
    "DriveSignal",
    "GaussianPulseDrive",
    "SinusoidalDrive",
    "StepDrive",
    "evaluate_drive",
]
