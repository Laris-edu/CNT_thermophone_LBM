"""Conservative 2:1 cell-centered merge/explode for an nx=1 open gas column.

Both levels use the existing D2Q37 collision and local transport mapping. Six
fine shadow rows carry populations only; they are neither collided nor physical
storage. An entire coarse step is transactional, including its two wall calls.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from boundary.planar_open import PreparedDiffuseWall, characteristic_top, crossing_moments
from core.collision_smrt import collide_fg
from core.macroscopic import heat_flux_lu, recover_macro
from core.planar_open_solver import OpenStepAudit
from core.solver import GasSolver2D
from core.streaming import pull_stream_open_y


@dataclass(frozen=True)
class FineWallAudit:
    """Actual interval exchanges; no whole-gas storage claim between syncs."""
    wall_mass_lu: float
    wall_energy_J_m2: float
    heat_time_s: float


@dataclass(frozen=True)
class InterfaceAudit:
    """SI [surface mass, x/y surface momentum, energy] entering each level."""
    fine: tuple[float, float, float, float]
    coarse: tuple[float, float, float, float]

    def as_array(self):
        fine, coarse = np.asarray(self.fine), np.asarray(self.coarse)
        return np.stack((fine, coarse, fine+coarse))


def two_level_layout(config):
    """Resolve exact shared cell faces; never snap a requested interface."""
    grid = config.get('grid', {})
    if not isinstance(grid,dict) or set(grid)!={'type','ratio','interface_m'}:
        raise ValueError('two-level grid accepts exactly type, ratio and interface_m')
    if grid.get('type') != 'two_level' or grid.get('ratio') != 2:
        raise ValueError("two-level grid requires type='two_level' and ratio=2")
    numerics = config.get('numerics', {})
    if numerics.get('nx') != 1:
        raise ValueError('two-level planar grid requires nx=1')
    h = float(config['lattice']['dx_m'])
    total = numerics.get('ny')
    if not np.isfinite(h) or h <= 0 or not isinstance(total, (int, np.integer)):
        raise ValueError('two-level layout requires positive dx and integer equivalent fine ny')
    position = float(grid.get('interface_m', np.nan))
    nf_real = position/h
    if not np.isfinite(nf_real):
        raise ValueError('two-level interface must be a finite common coarse/fine cell face')
    nf = int(round(nf_real))
    if abs(nf_real-nf) > 64*np.finfo(float).eps*max(1., abs(nf_real)) or nf % 2:
        raise ValueError('two-level interface must lie on a common coarse/fine cell face')
    if (total-nf) % 2:
        raise ValueError('two-level upper extent must contain an integer number of coarse cells')
    nc = (total-nf)//2
    if nf < 8 or nc < 8:
        raise ValueError('two-level open domain requires at least 8 physical cells per level')
    return dict(fine_cells=nf, coarse_cells=nc, shadow_cells=6,
                interface_m=nf*h, actual_length_m=(nf+2*nc)*h)


def _moments(f, g, lattice):
    sums = np.sum(f, axis=(0, 1))
    c2 = np.sum(lattice.c*lattice.c, axis=1)
    momentum = sums @ lattice.c
    return np.array([np.sum(sums), momentum[0], momentum[1],
                     .5*np.dot(sums, c2)+np.sum(g)])


class TwoLevelPlanarOpenSolver:
    """Lower fine and upper coarse physical blocks with a 2:1 time ratio.

    ``numerics.ny`` is the equivalent full fine-grid count. ``f``/``g`` are
    concatenated physical views by value, with explicit setters that split the
    levels; modify level arrays directly only for controlled initialization.
    """

    def __init__(self, config):
        self.config = config
        layout = two_level_layout(config)
        self.nf, self.nc = layout['fine_cells'], layout['coarse_cells']
        self.nx, self.ny, self.shadow_count = 1, self.nf+self.nc, 6
        self.actual_length_m = layout['actual_length_m']
        fine_config, coarse_config = deepcopy(config), deepcopy(config)
        fine_config['numerics']['ny'] = self.nf
        coarse_config['numerics']['ny'] = self.nc
        for name in ('dx_m', 'dt_s'):
            coarse_config['lattice'][name] *= 2
        self.fine_solver, self.coarse_solver = GasSolver2D(fine_config), GasSolver2D(coarse_config)
        self.mapping, self.coarse_mapping = self.fine_solver.mapping, self.coarse_solver.mapping
        if self.mapping.collision.bulk_viscosity_policy != 'diagnostic_zero':
            raise ValueError('two-level baseline supports only diagnostic_zero bulk viscosity')
        self.lattice, self.collision_backend = self.fine_solver.lattice, self.fine_solver.collision_backend
        h = self.mapping.lattice.dx_m
        self.cell_width_m = np.r_[np.full(self.nf, h), np.full(self.nc, 2*h)]
        self.z_m = np.r_[(np.arange(self.nf)+.5)*h, (self.nf+2*np.arange(self.nc)+1)*h]
        self.level_index = np.r_[np.zeros(self.nf, dtype=np.int8), np.ones(self.nc, dtype=np.int8)]
        self._weights = self.cell_width_m[:, None, None]/h
        self._si = np.array([self.mapping.rho_scale*h,
            self.mapping.rho_scale*self.mapping.velocity_scale*h,
            self.mapping.rho_scale*self.mapping.velocity_scale*h,
            self.mapping.pressure_scale*h])
        self._reset_audits()

    def _reset_audits(self):
        self.t_lu = 0
        self.last_audit = None
        self.last_substep_audits = ()
        self.last_interface_audit = None
        self.last_momentum_residual_kg_m_s = None
        self.level_steps = {'fine': 0, 'coarse': 0}
        self.cell_updates = 0

    @property
    def f(self):
        self._require_level_state()
        return np.concatenate((self.fine_solver.f, self.coarse_solver.f))

    @f.setter
    def f(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.ny, 1, self.lattice.q):
            raise ValueError('physical f shape must be (fine_cells+coarse_cells,1,37)')
        self.fine_solver.f, self.coarse_solver.f = values[:self.nf].copy(), values[self.nf:].copy()

    @property
    def g(self):
        self._require_level_state()
        return np.concatenate((self.fine_solver.g, self.coarse_solver.g))

    @g.setter
    def g(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.ny, 1, self.lattice.q):
            raise ValueError('physical g shape must be (fine_cells+coarse_cells,1,37)')
        self.fine_solver.g, self.coarse_solver.g = values[:self.nf].copy(), values[self.nf:].copy()

    def initialize_from_macro(self, rho, u, theta):
        rho = np.broadcast_to(np.asarray(rho, dtype=float), (self.ny, 1))
        theta = np.broadcast_to(np.asarray(theta, dtype=float), (self.ny, 1))
        u = np.broadcast_to(np.asarray(u, dtype=float), (self.ny, 1, 2))
        self.fine_solver.initialize_from_macro(rho[:self.nf], u[:self.nf], theta[:self.nf])
        self.coarse_solver.initialize_from_macro(rho[self.nf:], u[self.nf:], theta[self.nf:])
        self._reset_audits()

    def initialize_equilibrium(self):
        self.initialize_from_macro(self.mapping.lattice.rho_ref_lu, np.zeros(2), self.mapping.theta_ref_lu)

    def _require_level_state(self):
        if self.fine_solver.f is None or self.fine_solver.g is None or self.coarse_solver.f is None or self.coarse_solver.g is None:
            self.initialize_equilibrium()
        return self.fine_solver.f, self.fine_solver.g, self.coarse_solver.f, self.coarse_solver.g

    def _require_state(self):
        ff, gf, fc, gc = self._require_level_state()
        return np.concatenate((ff, fc)), np.concatenate((gf, gc))

    def checkpoint(self):
        """Own physical state until synchronized observations also commit.

        Shadow populations are local to step_wall_energy and need no snapshot.
        Copying both physical levels is intentional rollback overhead.
        """
        ff, gf, fc, gc = self._require_level_state()
        return dict(fine_f=ff.copy(), fine_g=gf.copy(), coarse_f=fc.copy(), coarse_g=gc.copy(),
            t_lu=self.t_lu, fine_t_lu=self.fine_solver.t_lu, coarse_t_lu=self.coarse_solver.t_lu,
            last_audit=deepcopy(self.last_audit), last_substep_audits=deepcopy(self.last_substep_audits),
            last_interface_audit=deepcopy(self.last_interface_audit),
            last_momentum_residual_kg_m_s=deepcopy(self.last_momentum_residual_kg_m_s),
            level_steps=self.level_steps.copy(), cell_updates=self.cell_updates)

    def restore(self, checkpoint):
        """Restore arrays, clocks and audits without sharing mutable snapshot data."""
        ff, gf, fc, gc = (checkpoint[name].copy()
                          for name in ('fine_f', 'fine_g', 'coarse_f', 'coarse_g'))
        self.fine_solver.f, self.fine_solver.g = ff, gf
        self.coarse_solver.f, self.coarse_solver.g = fc, gc
        self.t_lu = checkpoint['t_lu']
        self.fine_solver.t_lu = checkpoint['fine_t_lu']
        self.coarse_solver.t_lu = checkpoint['coarse_t_lu']
        self.last_audit = deepcopy(checkpoint['last_audit'])
        self.last_substep_audits = deepcopy(checkpoint['last_substep_audits'])
        self.last_interface_audit = deepcopy(checkpoint['last_interface_audit'])
        self.last_momentum_residual_kg_m_s = deepcopy(checkpoint['last_momentum_residual_kg_m_s'])
        self.level_steps = checkpoint['level_steps'].copy()
        self.cell_updates = checkpoint['cell_updates']

    def get_macro(self):
        f, g = self._require_state()
        return recover_macro(f, g, D=self.mapping.lattice.D, S=self.mapping.lattice.S, lattice=self.lattice)

    recover_macro = get_macro

    def get_pressure_lu(self):
        return self.get_macro().p

    def get_temperature_lu(self):
        return self.get_macro().theta

    def get_heat_flux_lu(self):
        ff, gf, fc, gc = self._require_level_state()
        return np.concatenate((heat_flux_lu(ff, gf, lattice=self.lattice, mapping=self.mapping),
            heat_flux_lu(fc, gc, lattice=self.lattice, mapping=self.coarse_mapping)))

    def _totals(self, f, g):
        moments = _moments(f*self._weights, g*self._weights, self.lattice)
        return float(moments[0]), float(moments[3])

    def conserved_totals_si(self):
        ff, gf, fc, gc = self._require_level_state()
        return (_moments(ff, gf, self.lattice)+2*_moments(fc, gc, self.lattice))*self._si

    def step(self, n_steps=1, boundary_callback=None):
        raise TypeError('use step_temperature or step_wall_energy for a synchronized coarse step')

    def step_temperature(self, temperature_K):
        return self.step_wall_energy(lambda exchange, index, midpoint:
                                    temperature_K(midpoint) if callable(temperature_K) else temperature_K)

    def step_wall_energy(self, solve_temperature):
        """Call ``solve_temperature(exchange_J_m2, index, midpoint_s)`` twice.

        All candidates, including coarse-boundary values, remain local until the
        second fine wall solve and synchronized budget/positivity checks succeed.
        Callback-owned external state must likewise be committed only on return.
        """
        ff, gf, fc, gc = self._require_level_state()
        old = _moments(ff, gf, self.lattice)+2*_moments(fc, gc, self.lattice)
        cp, cq = collide_fg(fc, gc, self.coarse_mapping, lattice=self.lattice, backend=self.collision_backend)
        cn, gn = pull_stream_open_y(cp, lattice=self.lattice), pull_stream_open_y(cq, lattice=self.lattice)
        characteristic_top(cp, cq, cn, gn, self.coarse_mapping, self.lattice)
        outlet = 2*(crossing_moments(cn, gn, self.lattice, top=True, incoming=True)
                    - crossing_moments(cp, cq, self.lattice, top=True))
        shadow_f = np.full((6, 1, self.lattice.q), np.nan)
        shadow_g = np.full_like(shadow_f, np.nan)
        for direction, cy in enumerate(self.lattice.c[:, 1].astype(int)):
            if cy < 0:
                shadow_f[:-2*cy, :, direction] = np.repeat(cp[:-cy, :, direction], 2, axis=0)
                shadow_g[:-2*cy, :, direction] = np.repeat(cq[:-cy, :, direction], 2, axis=0)
        interface_fine, walls = np.zeros(4), np.zeros(4)
        substeps = []
        for substep in range(2):
            fp, gp = collide_fg(ff, gf, self.mapping, lattice=self.lattice, backend=self.collision_backend)
            combined_f, combined_g = np.concatenate((fp, shadow_f)), np.concatenate((gp, shadow_g))
            streamed_f = pull_stream_open_y(combined_f, lattice=self.lattice)
            streamed_g = pull_stream_open_y(combined_g, lattice=self.lattice)
            fn, fn_g = streamed_f[:self.nf], streamed_g[:self.nf]
            wall = PreparedDiffuseWall(fp, gp, self.mapping, self.lattice)

            def exchange(temperature):
                return float(wall.energy_exchange_lu(temperature)*self._si[3])

            midpoint = (self.t_lu+substep+.5)*self.mapping.lattice.dt_s
            wall_temperature = solve_temperature(exchange, substep, midpoint)
            wall.write(wall_temperature, fn, fn_g)
            self._check_physical(fn, fn_g)
            wall_links = (crossing_moments(fn, fn_g, self.lattice, top=False, incoming=True)
                          - crossing_moments(fp, gp, self.lattice, top=False))
            interface_fine += (crossing_moments(fn, fn_g, self.lattice, top=True, incoming=True)
                               - crossing_moments(fp, gp, self.lattice, top=True))
            walls += wall_links
            substeps.append(FineWallAudit(float(wall_links[0]), float(wall_links[3]*self._si[3]), midpoint))
            ff, gf = fn, fn_g
            shadow_f, shadow_g = streamed_f[self.nf:], streamed_g[self.nf:]
        # Replace only links absent from coarse pull streaming, never its valid ones.
        for direction, cy in enumerate(self.lattice.c[:, 1].astype(int)):
            if cy > 0:
                cn[:cy, :, direction] = shadow_f[:2*cy, :, direction].reshape(cy, 2, 1).mean(axis=1)
                gn[:cy, :, direction] = shadow_g[:2*cy, :, direction].reshape(cy, 2, 1).mean(axis=1)
        self._check_physical(cn, gn)
        interface_coarse = 2*(crossing_moments(cn, gn, self.lattice, top=False, incoming=True)
                             - crossing_moments(cp, cq, self.lattice, top=False))
        new = _moments(ff, gf, self.lattice)+2*_moments(cn, gn, self.lattice)
        residual = new-old-walls-outlet
        # Sum the actual SI interval energies also exported for the two fine
        # solves, so downstream interval aggregation has precisely one meaning.
        wall_energy = sum(record.wall_energy_J_m2 for record in substeps)
        outlet_energy = float(outlet[3]*self._si[3])
        energy_residual = float((new[3]-old[3])*self._si[3]-wall_energy-outlet_energy)
        audit = OpenStepAudit(float(walls[0]), float(outlet[0]), wall_energy,
            outlet_energy, float(residual[0]), energy_residual)
        if (not np.isfinite(residual).all() or abs(audit.mass_residual_lu) > 1e-10
                or np.max(np.abs(residual[1:3])) > 1e-10
                or abs(audit.energy_residual_J_m2) > 1e-10):
            raise FloatingPointError('two-level synchronized mass/momentum/energy budget failed; state not committed')
        if np.max(np.abs(interface_fine+interface_coarse)) > 1e-10:
            raise FloatingPointError('two-level internal link cancellation failed; state not committed')
        # Atomic commit: nothing above writes physical solver-owned arrays or clocks.
        self.fine_solver.f, self.fine_solver.g = ff.copy(), gf.copy()
        self.coarse_solver.f, self.coarse_solver.g = cn, gn
        self.t_lu += 2
        self.fine_solver.t_lu = self.t_lu
        self.coarse_solver.t_lu = self.t_lu//2
        self.last_audit, self.last_substep_audits = audit, tuple(substeps)
        self.last_interface_audit = InterfaceAudit(tuple(interface_fine*self._si), tuple(interface_coarse*self._si))
        self.last_momentum_residual_kg_m_s = residual[1:3]*self._si[1:3]
        self.level_steps = {'fine': self.t_lu, 'coarse': self.t_lu//2}
        self.cell_updates += 2*self.nf+self.nc
        return audit

    @staticmethod
    def _check_physical(f, g):
        if not np.isfinite(f).all() or not np.isfinite(g).all():
            raise FloatingPointError('unfilled link or nonfinite two-level physical state')
        if np.min(f) <= 0 or np.min(g) <= 0:
            raise FloatingPointError('nonpositive two-level population; state not committed')
