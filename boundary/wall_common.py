"""Bottom-wall stencil helpers for boundary-aware D2Q37 thermal walls (P3-5+).

The Phase_2 solver streams fully periodically. For a bottom no-slip / Dirichlet wall
in the upper gas half-domain (wall plane at ``y = -1/2``, gas at rows ``0..ny-1``), the
populations with ``cy > 0`` that pull from ``y - cy < 0`` come from *below* the wall and
must be reconstructed instead of wrapped from the periodic top. This module only provides
lattice-derived indexing; the reconstruction rule lives in ``wall_thermal_abb.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.lattice import Lattice

BOTTOM_WALL_ROW = 0


@dataclass(frozen=True)
class BottomWallStencil:
    cy: np.ndarray            # integer wall-normal velocity component per direction
    incoming: np.ndarray      # direction indices with cy > 0 (pulled from below the wall)
    outgoing: np.ndarray      # direction indices with cy < 0
    grazing: np.ndarray       # direction indices with cy == 0
    opposite: np.ndarray      # opposite[a] for every direction a
    max_cy: int               # max(cy) -> deepest affected near-wall row + 1
    affected_rows: tuple[int, ...]  # rows whose incoming (cy>0) populations need the wall


def bottom_wall_stencil(lattice: Lattice) -> BottomWallStencil:
    cy = np.asarray(lattice.c[:, 1], dtype=int)
    opposite = np.asarray(lattice.opposite, dtype=int)
    max_cy = int(cy.max())
    return BottomWallStencil(
        cy=cy,
        incoming=np.where(cy > 0)[0],
        outgoing=np.where(cy < 0)[0],
        grazing=np.where(cy == 0)[0],
        opposite=opposite,
        max_cy=max_cy,
        affected_rows=tuple(range(max_cy)),
    )


def reflection_source_row(cy_a: int, target_row: int) -> int:
    """Halfway bounce-back source row for a wall plane at ``y = -1/2``.

    A population arriving at ``target_row`` with wall-normal speed ``cy_a > 0`` would have
    been pulled from ghost row ``target_row - cy_a < 0``. Reflecting that ghost row across
    the wall plane ``y = -1/2`` maps it to fluid row ``cy_a - 1 - target_row`` and reverses
    the direction (use ``opposite[a]`` there).
    """

    return int(cy_a) - 1 - int(target_row)


def pressure_preserving_rho(theta_wall_lu: float, mapping) -> float:
    """Wall density that keeps ``p = rho*theta`` at the reference value."""

    p_ref_lu = float(mapping.lattice.rho_ref_lu) * float(mapping.theta_ref_lu)
    return float(p_ref_lu / float(theta_wall_lu))
