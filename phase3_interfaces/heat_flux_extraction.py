"""Heat-flux extraction and conversion for Phase 3 handoff."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.lattice import make_lattice
from core.macroscopic import heat_flux_lu
from core.unit_mapping import (
    create_unit_mapping,
    heat_flux_lu_to_phys,
    heat_flux_phys_to_lu,
    pressure_lu_to_phys,
    temperature_lu_to_phys,
    temperature_phys_to_lu,
)


UPPER_GAS_WALL_NORMAL = np.array([0.0, 1.0])
HEAT_FLUX_SIGN_CONVENTION = "q_g''=-k_g*dT/dy|0+ is positive from film into upper gas"


def normal_heat_flux_lu(q_lu, wall_normal=UPPER_GAS_WALL_NORMAL):
    q = np.asarray(q_lu, dtype=float)
    normal = np.asarray(wall_normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    return np.sum(q * normal, axis=-1)


def extract_wall_heat_flux(
    f,
    g,
    wall_normal=UPPER_GAS_WALL_NORMAL,
    config: dict[str, Any] | None = None,
    return_physical: bool = False,
    velocity_set: str | None = None,
):
    """Conductive normal heat flux from ``f/g`` for the configured lattice.

    The lattice is taken from ``config`` (``lattice.velocity_set``) so D2Q37
    populations are handled correctly; an explicit ``velocity_set`` overrides it,
    and with neither it falls back to ``D2Q21`` for backward compatibility.
    Passing ``config`` also applies the conductive-flux moment/correction factors
    so the result matches ``GasSolver2D.get_heat_flux_lu`` (the Phase_3 handoff
    convention).
    """

    mapping = create_unit_mapping(config) if config is not None else None
    if velocity_set is None:
        velocity_set = mapping.lattice.velocity_set if mapping is not None else "D2Q21"
    lattice = make_lattice(velocity_set)
    q = heat_flux_lu(f, g, lattice=lattice, mapping=mapping)
    q_n = normal_heat_flux_lu(q, wall_normal)
    if return_physical:
        if mapping is None:
            raise ValueError("config is required for physical heat-flux conversion")
        return heat_flux_lu_to_phys(q_n, mapping)
    return q_n


def convert_heat_flux_lu_to_phys(q_lu, config: dict[str, Any]):
    return heat_flux_lu_to_phys(q_lu, create_unit_mapping(config))


def convert_heat_flux_phys_to_lu(q_phys, config: dict[str, Any]):
    return heat_flux_phys_to_lu(q_phys, create_unit_mapping(config))


def convert_temperature_phys_to_lu(T_phys, config: dict[str, Any]):
    return temperature_phys_to_lu(T_phys, create_unit_mapping(config))


def convert_pressure_lu_to_phys(p_lu, config: dict[str, Any]):
    return pressure_lu_to_phys(p_lu, create_unit_mapping(config))


def convert_temperature_lu_to_phys(theta_lu, config: dict[str, Any]):
    return temperature_lu_to_phys(theta_lu, create_unit_mapping(config))
