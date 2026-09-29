"""Exact same-grid factorization of the regularized open D2Q37 scheme.

Only nine moments persist between steps. Real populations are reconstructed
by the shared collision and streamed through the original boundary operators.
This does not implement independent thermal/acoustic grids.
"""
from copy import deepcopy

import numpy as np

from core.equilibrium import equilibrium_fg
from core.kinetic_moments import extract_moments, recover_moment_macro, collide_moments, heat_flux_moments
from core.planar_open_solver import PlanarOpenSolver


class MomentPlanarOpenSolver(PlanarOpenSolver):
    def __init__(self, config):
        if 'grid' in config:
            raise ValueError('moments9 currently supports the same uniform grid only')
        super().__init__(config)
        self.moments = None
        self._extrema = None

    def initialize_from_macro(self, rho, u, theta):
        shape = (self.ny, self.nx)
        f, g = equilibrium_fg(np.broadcast_to(rho, shape), np.broadcast_to(u, shape+(2,)),
            np.broadcast_to(theta, shape), self.mapping.lattice.S, self.lattice)
        self._commit_state(f, g)
        self.t_lu = 0
        self.last_audit = self.last_interface_audit = None

    def _require_moments(self):
        if self.moments is None:
            self.initialize_from_macro(self.mapping.lattice.rho_ref_lu, np.zeros(2), self.mapping.theta_ref_lu)
        return self.moments

    def _require_state(self):
        raise TypeError('moments9 has no persistent populations; use moments/get_macro')

    def _state_totals(self):
        z = self._require_moments()
        return float(np.sum(z[..., 0])/self.nx), float(np.sum(z[..., 3])/self.nx)

    def _collide_state(self):
        return collide_moments(self._require_moments(), self.mapping, self.lattice, self.collision_backend)

    def _commit_state(self, f, g):
        extrema = (float(np.min(f)), float(np.min(g)))
        if not np.isfinite(f).all() or not np.isfinite(g).all() or min(extrema) <= 0:
            raise FloatingPointError('nonfinite/nonpositive real populations; moment state not committed')
        z = extract_moments(f, g, self.lattice)
        if not np.isfinite(z).all():
            raise FloatingPointError('nonfinite moment state')
        self.moments, self._extrema = z, extrema

    def get_macro(self):
        return recover_moment_macro(self._require_moments(), self.mapping)

    def get_heat_flux_lu(self):
        return heat_flux_moments(self._require_moments(), self.mapping)

    def population_extrema(self):
        self._require_moments()
        return self._extrema

    def warmup_collision(self):
        collide_moments(self._require_moments()[:3], self.mapping, self.lattice, self.collision_backend)

    def checkpoint(self):
        return dict(moments=self._require_moments().copy(), extrema=self._extrema,
            t_lu=self.t_lu, last_audit=self.last_audit,
            last_interface_audit=deepcopy(self.last_interface_audit))

    def restore(self, checkpoint):
        self.moments = checkpoint['moments'].copy()
        self._extrema = checkpoint['extrema']
        self.t_lu, self.last_audit = checkpoint['t_lu'], checkpoint['last_audit']
        self.last_interface_audit = deepcopy(checkpoint['last_interface_audit'])
