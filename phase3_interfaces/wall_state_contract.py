"""Boundary-facing wall-state conversions for Phase 3."""

from __future__ import annotations

from typing import Any

from core.unit_mapping import create_unit_mapping, temperature_phys_to_lu


def wall_state_from_temperature(T_wall_phys, config: dict[str, Any]) -> dict[str, float]:
    mapping = create_unit_mapping(config)
    theta_wall_lu = float(temperature_phys_to_lu(T_wall_phys, mapping))
    return wall_state_from_theta(theta_wall_lu, config)


def wall_state_from_theta(theta_wall_lu, config: dict[str, Any]) -> dict[str, float]:
    mapping = create_unit_mapping(config)
    return {
        "theta_wall_lu": float(theta_wall_lu),
        "T_wall_K": float(theta_wall_lu) * mapping.temperature_scale,
        "u_wall_lu_x": 0.0,
        "u_wall_lu_y": 0.0,
        "normal_convention": "upper gas half-domain: wall_normal=+e_y, q_g''=-k_g*dT/dy|0+",
    }

