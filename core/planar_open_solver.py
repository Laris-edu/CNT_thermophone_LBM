"""给定壁温 -> 单侧空气 -> 直接压力的独立验证入口；不调用薄膜 ODE。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from boundary.planar_open import characteristic_top, crossing_totals, diffuse_temperature_wall
from core.collision_smrt import collide_fg
from core.solver import GasSolver2D
from core.streaming import pull_stream_open_y


@dataclass(frozen=True)
class OpenStepAudit:
    wall_mass_lu: float
    top_mass_lu: float
    wall_energy_J_m2: float
    top_energy_J_m2: float
    mass_residual_lu: float
    energy_residual_J_m2: float


class PlanarOpenSolver(GasSolver2D):
    """nx=1 平面波专用；边界定义独立于旧周期耦合入口。"""

    def __init__(self, config):
        super().__init__(config)
        if self.nx != 1 or self.ny < 8:
            raise ValueError("planar open validation requires nx=1 and ny>=8")
        self.last_audit = None

    @property
    def z_m(self):
        return (np.arange(self.ny) + .5) * self.mapping.lattice.dx_m

    def _totals(self, f, g):
        c2 = np.sum(self.lattice.c**2, axis=1)
        return float(np.sum(f)/self.nx), float(np.sum(.5*f*c2+g)/self.nx)

    def step(self, n_steps=1, boundary_callback=None):
        raise TypeError("use step_temperature(temperature_K); no film ODE or periodic callback")

    def step_temperature(self, temperature_K):
        """指定本时间区间中点的壁温；交换量属于 [t_n,t_n+dt]。"""
        f, g = self._require_state()
        mass0, energy0 = self._totals(f, g)
        fp, gp = collide_fg(f, g, self.mapping, lattice=self.lattice)
        fn = pull_stream_open_y(fp, lattice=self.lattice)
        gn = pull_stream_open_y(gp, lattice=self.lattice)
        diffuse_temperature_wall(fp, gp, fn, gn, self.mapping, self.lattice, temperature_K)
        characteristic_top(fp, gp, fn, gn, self.mapping, self.lattice)
        if not np.isfinite(fn).all() or not np.isfinite(gn).all():
            raise FloatingPointError("unfilled link or nonfinite open-domain state")
        exchanges = []
        for top in (False, True):
            before = np.asarray(crossing_totals(fp, gp, self.lattice, top=top))
            after = np.asarray(crossing_totals(fn, gn, self.lattice, top=top, incoming=True))
            exchanges.append(after-before)
        wall, outlet = exchanges
        mass1, energy1 = self._totals(fn, gn)
        scale = self.mapping.pressure_scale * self.mapping.lattice.dx_m
        self.last_audit = OpenStepAudit(float(wall[0]), float(outlet[0]),
            float(wall[1]*scale), float(outlet[1]*scale),
            float(mass1-mass0-wall[0]-outlet[0]),
            float((energy1-energy0-wall[1]-outlet[1])*scale))
        self.f, self.g = fn, gn
        self.t_lu += 1
        return self.last_audit
