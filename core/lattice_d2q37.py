"""Candidate D2Q37 multispeed lattice for the Phase 2 fallback route.

This module starts the D2Q37 / equivalent ninth-order velocity-set path after
the D2Q21 physical-timestep high-mode failure.  It is wired through the Phase 2
lattice-family registry for diagnostic runs, but it is not a production
replacement until dynamic transport validation passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


THETA_Q_D2Q37 = 0.6979533220196852


@dataclass(frozen=True)
class LatticeD2Q37:
    c: np.ndarray
    w: np.ndarray
    theta_q: float
    opposite: np.ndarray
    q: int
    d: int


def _shell_velocities() -> list[tuple[list[tuple[int, int]], float]]:
    return [
        ([(0, 0)], 0.2331506691323516),
        ([(1, 0), (-1, 0), (0, 1), (0, -1)], 0.10730609154221887),
        ([(1, 1), (1, -1), (-1, 1), (-1, -1)], 0.05766785988879502),
        ([(2, 0), (-2, 0), (0, 2), (0, -2)], 0.01420821615845085),
        (
            [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
            0.005353049000513803,
        ),
        ([(2, 2), (2, -2), (-2, 2), (-2, -2)], 0.0010119375926735928),
        ([(3, 0), (-3, 0), (0, 3), (0, -3)], 0.00024530102775772127),
        (
            [(3, 1), (3, -1), (-3, 1), (-3, -1), (1, 3), (1, -3), (-1, 3), (-1, -3)],
            0.00028341425299420404,
        ),
    ]


def make_d2q37(dtype=np.float64) -> LatticeD2Q37:
    """Return the candidate positive-weight D2Q37 lattice."""

    velocities: list[tuple[int, int]] = []
    weights: list[float] = []
    for shell, weight in _shell_velocities():
        velocities.extend(shell)
        weights.extend([weight] * len(shell))
    c = np.asarray(velocities, dtype=dtype)
    w = np.asarray(weights, dtype=dtype)
    index = {tuple(item): i for i, item in enumerate(velocities)}
    opposite = np.asarray([index[(-cx, -cy)] for cx, cy in velocities], dtype=np.int64)
    return LatticeD2Q37(c=c, w=w, theta_q=THETA_Q_D2Q37, opposite=opposite, q=37, d=2)


def moment(exponents: tuple[int, int], lattice: LatticeD2Q37 | None = None) -> float:
    lattice = lattice or make_d2q37()
    m, n = exponents
    return float(np.sum(lattice.w * lattice.c[:, 0] ** m * lattice.c[:, 1] ** n))


def _normal_even_moment(order: int, theta: float) -> float:
    if order == 0:
        return 1.0
    value = 1.0
    for item in range(1, order, 2):
        value *= item
    return value * theta ** (order // 2)


def _gaussian_target(exponents: tuple[int, int], theta: float) -> float:
    m, n = exponents
    if m % 2 or n % 2:
        return 0.0
    return _normal_even_moment(m, theta) * _normal_even_moment(n, theta)


def assert_d2q37_moments(tol: float = 1.0e-12) -> None:
    """Validate the candidate ninth-order symmetry / eighth-order even moments."""

    lattice = make_d2q37()
    if lattice.q != 37 or lattice.c.shape != (37, 2) or lattice.w.shape != (37,):
        raise AssertionError("D2Q37 layout is invalid")
    if np.min(lattice.w) <= 0.0:
        raise AssertionError("D2Q37 weights must be positive")
    for total in range(1, 10, 2):
        for m in range(total + 1):
            n = total - m
            value = moment((m, n), lattice)
            if abs(value) > tol:
                raise AssertionError(f"M{m}{n}: odd moment {value:.17g} exceeds tolerance")
    for m, n in product(range(9), repeat=2):
        total = m + n
        if total <= 8 and total % 2 == 0:
            value = moment((m, n), lattice)
            target = _gaussian_target((m, n), lattice.theta_q)
            if abs(value - target) > tol:
                raise AssertionError(
                    f"M{m}{n}: got {value:.17g}, expected {target:.17g}, "
                    f"err={abs(value - target):.3e}"
                )
    for i, opp in enumerate(lattice.opposite):
        if not np.allclose(lattice.c[i] + lattice.c[opp], 0.0, atol=tol, rtol=0.0):
            raise AssertionError(f"opposite[{i}]={opp} is not the negative velocity")
