"""单侧开放平面气体推进；支持给温或由实际壁面交换确定壁温。"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import numpy as np

from boundary.planar_open import PreparedDiffuseWall, characteristic_top, crossing_totals, crossing_moments
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
        self.interface_face_index = None
        self.last_interface_audit = None

    @property
    def z_m(self):
        return (np.arange(self.ny) + .5) * self.mapping.lattice.dx_m

    def _totals(self, f, g):
        c2 = np.sum(self.lattice.c**2, axis=1)
        return float(np.sum(f)/self.nx), float(np.sum(.5*f*c2+g)/self.nx)

    def _state_totals(self):
        return self._totals(*self._require_state())

    def _collide_state(self):
        return collide_fg(*self._require_state(), self.mapping,
                          lattice=self.lattice, backend=self.collision_backend)

    def _commit_state(self, f, g):
        self.f, self.g = f, g

    def population_extrema(self):
        return tuple(float(np.min(v)) for v in self._require_state())

    def warmup_collision(self):
        f, g = self._require_state()
        collide_fg(f[:3], g[:3], self.mapping, lattice=self.lattice, backend=self.collision_backend)

    def checkpoint(self):
        """Owned snapshot for a synchronized pair of open-domain steps."""
        f, g = self._require_state()
        return dict(f=f.copy(), g=g.copy(), t_lu=self.t_lu,
                    last_audit=self.last_audit, last_interface_audit=deepcopy(self.last_interface_audit))

    def restore(self, checkpoint):
        self.f, self.g = checkpoint['f'].copy(), checkpoint['g'].copy()
        self.t_lu = checkpoint['t_lu']
        self.last_audit = checkpoint['last_audit']
        self.last_interface_audit = deepcopy(checkpoint['last_interface_audit'])

    def step(self, n_steps=1, boundary_callback=None):
        raise TypeError("use step_temperature(temperature_K); no film ODE or periodic callback")

    def step_temperature(self, temperature_K):
        """指定本时间区间中点的壁温；交换量属于 [t_n,t_n+dt]。"""
        return self.step_wall_energy(lambda exchange: temperature_K)

    def step_wall_energy(self, solve_temperature):
        """Call solve_temperature(exchange_J_m2) to obtain the midpoint wall T.

        Trial exchange evaluations construct only incoming wall populations,
        without writing the streamed state. Commit once, after the callback and all
        boundary checks succeed. The gas layer has no knowledge of a film ODE.
        """
        mass0, energy0 = self._state_totals()
        fp, gp = self._collide_state()
        fn = pull_stream_open_y(fp, lattice=self.lattice)
        gn = pull_stream_open_y(gp, lattice=self.lattice)
        scale = self.mapping.pressure_scale * self.mapping.lattice.dx_m
        wall = PreparedDiffuseWall(fp, gp, self.mapping, self.lattice)

        def exchange(temperature):
            return float(wall.energy_exchange_lu(temperature)*scale)

        temperature_K = solve_temperature(exchange)
        # The last trial need not be the returned root.
        wall.write(temperature_K, fn, gn)
        characteristic_top(fp, gp, fn, gn, self.mapping, self.lattice)
        if not np.isfinite(fn).all() or not np.isfinite(gn).all():
            raise FloatingPointError("unfilled link or nonfinite open-domain state")
        if np.min(fn) <= 0 or np.min(gn) <= 0:
            raise FloatingPointError("nonpositive open-domain population; state not committed")
        exchanges = []
        for top in (False, True):
            before = np.asarray(crossing_totals(fp, gp, self.lattice, top=top))
            after = np.asarray(crossing_totals(fn, gn, self.lattice, top=top, incoming=True))
            exchanges.append(after-before)
        wall, outlet = exchanges
        interface = None
        if self.interface_face_index is not None:
            face = self.interface_face_index
            if type(face) is not int or not 3 <= face <= self.ny-3:
                raise ValueError('interface_face_index must be an interior face with three rows each side')
            lower = (crossing_moments(fn[:face],gn[:face],self.lattice,top=True,incoming=True)
                     - crossing_moments(fp[:face],gp[:face],self.lattice,top=True))
            upper = (crossing_moments(fn[face:],gn[face:],self.lattice,top=False,incoming=True)
                     - crossing_moments(fp[face:],gp[face:],self.lattice,top=False))
            m = self.mapping
            units = m.lattice.dx_m*np.array([m.rho_scale,m.rho_scale*m.velocity_scale,
                                              m.rho_scale*m.velocity_scale,m.pressure_scale])
            interface = np.array([lower,upper,lower+upper])*units
        mass1, energy1 = self._totals(fn, gn)
        audit = OpenStepAudit(float(wall[0]), float(outlet[0]),
            float(wall[1]*scale), float(outlet[1]*scale),
            float(mass1-mass0-wall[0]-outlet[0]),
            float((energy1-energy0-wall[1]-outlet[1])*scale))
        self._commit_state(fn, gn)
        self.last_audit = audit
        self.last_interface_audit = interface
        self.t_lu += 1
        return self.last_audit
