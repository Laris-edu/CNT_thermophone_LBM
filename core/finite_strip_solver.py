"""Finite-width strip in an upper-half gas domain (invariant out of plane).

Reuses the common D2Q37 collision and SI/LU mapping. The x/top characteristic
boundary and optional reference-state sponge are experimental; numerical
completion and energy closure do not certify small acoustic reflection.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from boundary.finite_strip import (FiniteStripLinks, PreparedStripWall,
                                   characteristic_outer, pull_stream_open_xy)
from core.collision_smrt import collide_fg
from core.equilibrium import equilibrium_fg
from core.solver import GasSolver2D
from core.array_backend import to_numpy


@dataclass(frozen=True)
class FiniteStripAudit:
    """All signed exchanges are positive into the upper gas domain.

    Wall arrays are per film bin. Whole-domain energies are J/m of invariant
    out-of-plane length; mass_lu denotes a sum of discrete cell populations.
    """
    wall_mass_lu: np.ndarray
    wall_energy_J_m2: np.ndarray
    outer_mass_lu: float
    outer_energy_J_m: float
    sponge_mass_lu: float
    sponge_energy_J_m: float
    gas_energy_before_J_m: float
    gas_energy_after_J_m: float
    mass_residual_lu: float
    energy_residual_J_m: float
    pml_mass_lu: float = 0.
    pml_energy_J_m: float = 0.
    gas_mass_after_lu: float = float('nan')
    minimum_f: float = float('nan')
    minimum_g: float = float('nan')


class FiniteStripSolver(GasSolver2D):
    """Mixed diffuse/specular bottom, open nonperiodic x/top, no film ODE."""

    reflection_validation = 'not_run'
    boundary_model = 'experimental_local_normal_characteristic_with_reference_sponge'
    exchange_is_local = True
    supports_gpu = True

    def __init__(self, config, *, film_width_m=1e-3, x_min_m=None,
                 film_center_m=0., sponge_cells=0, sponge_rate=0.,
                 pml_cells=0, pml_target_return=.001, pml_alpha_Hz=1000.):
        super().__init__(config)
        if min(self.nx, self.ny) < 8:
            raise ValueError('finite strip requires nx,ny>=8')
        dx = self.mapping.lattice.dx_m
        if not np.isfinite(film_width_m) or film_width_m < 0 or not np.isfinite(film_center_m):
            raise ValueError('finite film width>=0 and centre required')
        self.x_min_m = -.5*self.nx*dx if x_min_m is None else float(x_min_m)
        if not np.isfinite(self.x_min_m):
            raise ValueError('finite x_min_m required')
        self.requested_film_width_m = float(film_width_m)
        self.requested_film_center_m = float(film_center_m)
        # The continuum film is discretized by complete wall bins. Record both
        # requested and actual faces; callers must retain the latter in outputs.
        start = int(round((film_center_m-film_width_m/2-self.x_min_m)/dx))
        stop = int(round((film_center_m+film_width_m/2-self.x_min_m)/dx))
        if film_width_m > 0 and stop <= start:
            raise ValueError('film width resolves fewer than one bin')
        if film_width_m > 0 and (start < 3 or stop > self.nx-3):
            raise ValueError('film must remain at least three cells from open sides')
        self.boundary = FiniteStripLinks(self.ny, self.nx, self.lattice,
                                         film_start=start, film_stop=stop)
        self.film_dx_m = dx
        self.film_x_min_m, self.film_x_max_m = self.x_min_m+np.array([start, stop])*dx
        self.film_x_m = self.x_min_m+(np.arange(start, stop)+.5)*dx
        if type(sponge_cells) is not int or not 0 <= sponge_cells <= min(self.ny, self.nx//2):
            raise ValueError('sponge_cells must be an integer within the domain')
        if not np.isfinite(sponge_rate) or not 0 <= sponge_rate < 1:
            raise ValueError('sponge_rate must be finite in [0,1) per step')
        if sponge_rate > 0 and sponge_cells == 0:
            raise ValueError('positive sponge_rate requires sponge_cells>0')
        self.sponge_cells, self.sponge_rate = sponge_cells, float(sponge_rate)
        self._sponge = np.zeros((self.ny, self.nx))
        if sponge_cells:
            x, y = np.arange(self.nx)+.5, np.arange(self.ny)+.5
            distance = np.minimum(np.minimum(x, self.nx-x)[None, :], (self.ny-y)[:, None])
            self._sponge = sponge_rate*np.maximum(0., 1-distance/sponge_cells)**2
        self._reference_f, self._reference_g = equilibrium_fg(self.mapping.lattice.rho_ref_lu,
            np.zeros(2), self.mapping.theta_ref_lu, self.mapping.lattice.S, self.lattice)
        self._kinetic = .5*np.sum(self.lattice.c**2, axis=1)
        self.pml = None
        if pml_cells:
            if sponge_rate:
                raise ValueError('acoustic PML and legacy sponge cannot be combined')
            from boundary.acoustic_pml import AcousticPML
            self.pml = AcousticPML(self.mapping,self.lattice,self.ny,self.nx,
                cells=pml_cells,target_return=pml_target_return,alpha_Hz=pml_alpha_Hz,
                backend=self.collision_backend)
            self.boundary_model = 'experimental_acoustic_cfs_pml_with_outer_characteristic'
        self.last_audit = None
        self.xp = np
        if self.collision_backend == 'cupy':
            import cupy as cp
            self.xp = cp
            self.boundary.to_device()
            self._sponge = cp.asarray(self._sponge)
            self._reference_f = cp.asarray(self._reference_f)
            self._reference_g = cp.asarray(self._reference_g)
            self._kinetic = cp.asarray(self._kinetic)

    def initialize_from_macro(self, rho, u, theta):
        if self.collision_backend == 'cupy':
            from core.collision_gpu import equilibrium_gpu
            xp = self.xp
            rho = xp.broadcast_to(xp.asarray(rho, dtype=float), (self.ny, self.nx))
            theta = xp.broadcast_to(xp.asarray(theta, dtype=float), (self.ny, self.nx))
            u = xp.broadcast_to(xp.asarray(u, dtype=float), (self.ny, self.nx, 2))
            self.f, self.g = equilibrium_gpu(rho, u, theta, self.mapping.lattice.S, self.lattice)
            self.t_lu = 0
        else:
            super().initialize_from_macro(rho,u,theta)
        if self.pml is not None:
            self.pml.memory.fill(0.)
        self.last_audit = None

    def get_macro(self):
        if self.collision_backend == 'cupy':
            from core.collision_gpu import recover_macro_gpu
            return recover_macro_gpu(*self._require_state(), D=self.mapping.lattice.D,
                                     S=self.mapping.lattice.S, lattice=self.lattice)
        return super().get_macro()

    @property
    def x_m(self):
        return self.x_min_m+(np.arange(self.nx)+.5)*self.mapping.lattice.dx_m

    @property
    def y_m(self):
        return (np.arange(self.ny)+.5)*self.mapping.lattice.dx_m

    def _totals(self, f, g):
        if self.collision_backend in ('numba', 'cupy'):
            return np.asarray(self._population_statistics(f, g)[:2])
        return np.array([np.sum(f), np.sum(f*self._kinetic+g)])

    def _population_statistics(self, f, g):
        if self.collision_backend == 'cupy':
            from core.finite_strip_gpu import population_statistics
            return population_statistics(f, g, self._kinetic)
        if self.collision_backend == 'numba':
            from core.finite_strip_numba import population_statistics
            return population_statistics(f, g, self._kinetic)
        return (float(np.sum(f)), float(np.sum(f*self._kinetic+g)),
                float(np.min(f)), float(np.min(g)),
                bool(np.isfinite(f).all() and np.isfinite(g).all()))

    def population_extrema(self):
        return tuple(float(self.xp.min(v)) for v in self._require_state())

    def warmup_collision(self):
        f, g = self._require_state()
        collide_fg(f[:3, :3], g[:3, :3], self.mapping, lattice=self.lattice,
                   backend=self.collision_backend)

    def step(self, n_steps=1, boundary_callback=None):
        raise TypeError('use step_temperature or step_wall_energy; periodic step is disabled')

    def step_temperature(self, temperature_K):
        return self.step_wall_energy(lambda exchange: temperature_K)

    def step_wall_energy(self, solve_temperature):
        """Ask callback(exchange) for per-bin midpoint temperatures in K.

        exchange(T) returns each bin's actual single-side J/m² during this step.
        Trials are strictly local in T and do not modify the gas state. Commit
        only after all boundary operations and population checks succeed.
        """
        f, g = self._require_state()
        before = self._totals(f, g)
        pml_change = np.zeros(2)
        if self.pml is not None:
            f,g,pml_memory,pml_change = self.pml.advance(f,g,self.pml.memory,.5)
            pml_change = to_numpy(pml_change)
            self._check_populations(f,g)
        fp, gp = collide_fg(f, g, self.mapping, lattice=self.lattice, backend=self.collision_backend)
        if self.collision_backend == 'cupy':
            from core.finite_strip_gpu import stream_open_xy_pair
            fn, gn = stream_open_xy_pair(fp, gp, self.lattice.c)
        elif self.collision_backend == 'numba':
            from core.finite_strip_numba import stream_open_xy_pair
            fn, gn = stream_open_xy_pair(fp, gp, self.lattice.c)
        else:
            fn, gn = (pull_stream_open_xy(v, lattice=self.lattice) for v in (fp, gp))
        self.boundary.write_specular(fp, gp, fn, gn)
        wall = PreparedStripWall(fp, gp, self.mapping, self.lattice, self.boundary)
        dx = self.mapping.lattice.dx_m
        area_scale = self.mapping.pressure_scale*dx
        length_scale = area_scale*dx

        def exchange(temperature_K):
            return to_numpy(wall.energy_exchange_lu(temperature_K))*area_scale

        temperature_K = solve_temperature(exchange)
        wall.write(temperature_K, fn, gn)
        characteristic_outer(fp, gp, fn, gn, self.mapping, self.lattice, self.boundary)
        after_statistics = self._check_populations(fn, gn)
        # Independent accounting reads actual written populations, not the trial
        # heat function or film equation. Count every corner link exactly once.
        wall_change = (self.boundary.totals(fn, gn, wall=True, incoming=True)
                       -self.boundary.totals(fp, gp, wall=True, incoming=False))
        outer_change = (self.boundary.totals(fn, gn, wall=False, incoming=True)
                        -self.boundary.totals(fp, gp, wall=False, incoming=False))
        wall_change, outer_change = to_numpy(wall_change), to_numpy(outer_change)
        sponge_change = np.zeros(2)
        if self.sponge_rate:
            pre_sponge = np.asarray(after_statistics[:2])
            weight = self._sponge[..., None]
            fn += weight*(self._reference_f-fn)
            gn += weight*(self._reference_g-gn)
            after_statistics = self._check_populations(fn, gn)
            sponge_change = np.asarray(after_statistics[:2])-pre_sponge
        if self.pml is not None:
            fn,gn,pml_memory,change = self.pml.advance(fn,gn,pml_memory,.5)
            pml_change += to_numpy(change)
            after_statistics = self._check_populations(fn,gn)
        after = np.asarray(after_statistics[:2])
        residual = after-before-wall_change.sum(axis=1)-outer_change-sponge_change-pml_change
        audit = FiniteStripAudit(wall_change[0].copy(), wall_change[1]*area_scale,
            float(outer_change[0]), float(outer_change[1]*length_scale),
            float(sponge_change[0]), float(sponge_change[1]*length_scale),
            float(before[1]*length_scale), float(after[1]*length_scale),
            float(residual[0]), float(residual[1]*length_scale),
            float(pml_change[0]),float(pml_change[1]*length_scale),
            gas_mass_after_lu=float(after[0]),minimum_f=after_statistics[2],minimum_g=after_statistics[3])
        self.f, self.g = fn, gn
        if self.pml is not None:
            self.pml.memory = pml_memory
        self.last_audit = audit
        self.t_lu += 1
        return audit

    def _check_populations(self, f, g):
        statistics = self._population_statistics(f, g)
        if not statistics[4]:
            raise FloatingPointError('unfilled link or nonfinite finite-strip state; state not committed')
        if min(statistics[2:4]) <= 0:
            raise FloatingPointError('nonpositive finite-strip population; state not committed')
        return statistics
