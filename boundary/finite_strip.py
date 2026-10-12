"""Half-cell finite strip and experimental open x/top link boundaries.

The strip is a zero-thickness wall at y=0. Outside it, y=0 is a gas
symmetry plane. Links are assigned by their geometric intersection, not
their destination column. No x or y periodic wrap is used.

The outer boundary applies a *local normal* linear acoustic characteristic
projection. Together with an explicitly audited sponge it is an experimental
absorber, not a validated multidimensional non-reflecting condition.
"""
from __future__ import annotations

import numpy as np

from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro


def pull_stream_open_xy(distribution, *, lattice):
    """Stream only in-domain links; every missing incoming link remains NaN."""
    dist = np.asarray(distribution, dtype=float)
    if dist.ndim != 3 or dist.shape[-1] != lattice.q:
        raise ValueError('open xy streaming requires (ny,nx,Q)')
    ny, nx = dist.shape[:2]
    out = np.full_like(dist, np.nan)
    for a, (cx, cy) in enumerate(lattice.c.astype(int)):
        x0, x1 = max(0, cx), min(nx, nx+cx)
        y0, y1 = max(0, cy), min(ny, ny+cy)
        if x0 < x1 and y0 < y1:
            out[y0:y1, x0:x1, a] = dist[y0-cy:y1-cy, x0-cx:x1-cx, a]
    return out


def _crossing_indices(ny, nx, c, *, incoming):
    """One entry per boundary-crossing population, including corner links."""
    rows, columns, directions = [], [], []
    for a, velocity in enumerate(c):
        cx, cy = (-velocity if incoming else velocity)
        # Invalid x endpoints first, then invalid y endpoints with valid x.
        xa, xb = max(0, -cx), min(nx, nx-cx)
        for ys, xs in ((np.arange(ny), np.r_[np.arange(xa), np.arange(xb, nx)]),
                       (np.r_[np.arange(max(0, -cy)), np.arange(min(ny, ny-cy), ny)],
                        np.arange(xa, xb))):
            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            rows.extend(yy.ravel())
            columns.extend(xx.ravel())
            directions.extend([a]*yy.size)
    return tuple(np.asarray(v, dtype=np.intp) for v in (rows, columns, directions))


def _select(indices, mask):
    return tuple(v[mask] for v in indices)


class FiniteStripLinks:
    """Precomputed boundary topology. Film bounds are grid-face indices."""

    def __init__(self, ny, nx, lattice, *, film_start, film_stop):
        self.ny, self.nx, self.lattice = ny, nx, lattice
        self.film_start, self.film_stop = int(film_start), int(film_stop)
        self.nfilm = self.film_stop-self.film_start
        if not 0 <= self.film_start <= self.film_stop <= nx:
            raise ValueError('film faces must lie inside the x domain')
        c = lattice.c.astype(int)
        reverse_y = np.array([np.flatnonzero(np.all(c == (cx, -cy), axis=1))[0]
                              for cx, cy in c], dtype=np.intp)
        for incoming in (False, True):
            indices = _crossing_indices(ny, nx, c, incoming=incoming)
            y, x, a = indices
            cx, cy = c[a].T
            sign = -1 if incoming else 1
            end_x, end_y = x+sign*cx, y+sign*cy
            bottom = end_y < 0
            hit_x = np.full(x.shape, np.nan)
            # Parametric ray-plane intersection in cell-width coordinates.
            hit_x[bottom] = x[bottom]+.5-cx[bottom]*(y[bottom]+.5)/cy[bottom]
            # A ray landing exactly on a membrane endpoint belongs to that
            # endpoint on BOTH sides. Half-open endpoint classification would
            # spuriously break x-reflection symmetry for multispeed diagonal rays.
            film = bottom & (self.nfilm > 0) & (hit_x >= film_start) & (hit_x <= film_stop)
            wall_x = hit_x[film]
            bins = np.floor(wall_x).astype(np.intp)-film_start
            # At an interior bin face choose by the upward-oriented ray. This
            # keeps both time-reversed links in the same bin while preserving
            # left/right reflection. Endpoints belong to their edge bin.
            upward_cx = cx[film] if incoming else -cx[film]
            on_face = wall_x == np.floor(wall_x)
            bins -= on_face & (upward_cx > 0)
            bins = np.clip(bins, 0, max(0, self.nfilm-1))
            reflected_inside = bottom & ~film & (end_x >= 0) & (end_x < nx)
            outer = ~film & ~reflected_inside
            if incoming:
                self.wall_in, self.wall_bin_in = _select(indices, film), bins
                self.outer_in = _select(indices, outer)
                self.specular_in = _select(indices, reflected_inside)
                self.specular_source = (cy[reflected_inside]-y[reflected_inside]-1,
                                        end_x[reflected_inside], reverse_y[a[reflected_inside]])
            else:
                self.wall_out, self.wall_bin_out = _select(indices, film), bins
                self.outer_out = _select(indices, outer)
        self._prepare_outer_faces(c)

    def _prepare_outer_faces(self, c):
        y, x, a = self.outer_in
        cx, cy = c[a].T
        # Trace backward from each incoming cell to the first exterior face.
        # abs(y_hit) accounts for a prior reflection on the bottom symmetry plane.
        times = np.full((3, len(a)), np.inf)
        left, right, top = cx > 0, cx < 0, cy < 0
        times[0, left] = (x[left]+.5)/cx[left]
        times[1, right] = (self.nx-x[right]-.5)/(-cx[right])
        times[2, top] = (self.ny-y[top]-.5)/(-cy[top])
        self.outer_face = np.argmin(times, axis=0)
        t = times[self.outer_face, np.arange(len(a))]
        if np.any(t > 1+1e-12):
            raise ValueError('outer incoming ray did not intersect an exterior face')
        raw_y = y+.5-cy*t
        hit_y = np.abs(raw_y)
        hit_x = x+.5-cx*t
        self.outer_direction = a.copy()
        reflected = raw_y < 0
        for direction in np.unique(a[reflected]):
            reflected_direction = np.flatnonzero(np.all(c == (c[direction, 0], -c[direction, 1]), axis=1))[0]
            self.outer_direction[reflected & (a == direction)] = reflected_direction
        # Interpolate at the geometric face intersection. Nearest floor-cell
        # sampling gives unequal choices at exact left/right mirror face hits.
        coordinate = np.where(self.outer_face == 2, hit_x, hit_y)-.5
        extent = np.where(self.outer_face == 2, self.nx, self.ny)
        coordinate = np.maximum(0., np.minimum(coordinate, extent-1))
        self.outer_tangent = np.floor(coordinate).astype(np.intp)
        self.outer_tangent_next = np.minimum(self.outer_tangent+1, extent-1)
        self.outer_fraction = coordinate-self.outer_tangent

    def write_specular(self, f_post, g_post, f_new, g_new):
        f_new[self.specular_in] = f_post[self.specular_source]
        g_new[self.specular_in] = g_post[self.specular_source]

    def totals(self, f, g, *, wall, incoming):
        """Actual link mass and energy, without continuum moment estimates."""
        indices = (self.wall_in if incoming else self.wall_out) if wall else (
            self.outer_in if incoming else self.outer_out)
        fl, gl = f[indices], g[indices]
        kinetic = .5*np.sum(self.lattice.c[indices[2]]**2, axis=1)
        if wall:
            bins = self.wall_bin_in if incoming else self.wall_bin_out
            return np.array([np.bincount(bins, weights=fl, minlength=self.nfilm),
                np.bincount(bins, weights=kinetic*fl+gl, minlength=self.nfilm)])
        return np.array([np.sum(fl), np.sum(kinetic*fl+gl)])


class PreparedStripWall:
    """Local-in-bin diffuse wall; trial evaluations never mutate the gas.

    Bin density is set from equality of actual incoming/outgoing discrete mass.
    The sole thermal parameter for bin b is temperature_K[b].
    """

    def __init__(self, f_post, g_post, mapping, lattice, links):
        self.mapping, self.lattice, self.links = mapping, lattice, links
        self.outgoing = links.totals(f_post, g_post, wall=True, incoming=False)
        self._temperature = None

    def _incoming(self, temperature_K):
        temperature = np.broadcast_to(np.asarray(temperature_K, dtype=float), (self.links.nfilm,))
        if not np.isfinite(temperature).all() or np.any(temperature <= 0):
            raise ValueError('wall temperatures must be finite and positive')
        if self._temperature is None or not np.array_equal(temperature, self._temperature):
            count = self.links.nfilm
            if count == 0:
                self._f = self._g = np.empty(0)
            else:
                fw, gw = equilibrium_fg(np.ones(count), np.zeros((count, 2)),
                    temperature/self.mapping.temperature_scale, self.mapping.lattice.S, self.lattice)
                bins, directions = self.links.wall_bin_in, self.links.wall_in[2]
                fl, gl = fw[bins, directions], gw[bins, directions]
                denominator = np.bincount(bins, weights=fl, minlength=count)
                if np.any(denominator <= 0):
                    raise FloatingPointError('film bin has no positive incoming mass')
                rho_wall = self.outgoing[0]/denominator
                self._f, self._g = fl*rho_wall[bins], gl*rho_wall[bins]
            self._temperature = temperature.copy()
        return self._f, self._g

    def energy_exchange_lu(self, temperature_K):
        fl, gl = self._incoming(temperature_K)
        kinetic = .5*np.sum(self.lattice.c[self.links.wall_in[2]]**2, axis=1)
        return np.bincount(self.links.wall_bin_in, weights=kinetic*fl+gl,
                           minlength=self.links.nfilm)-self.outgoing[1]

    def write(self, temperature_K, f_new, g_new):
        f_new[self.links.wall_in], g_new[self.links.wall_in] = self._incoming(temperature_K)


def characteristic_outer(f_post, g_post, f_new, g_new, mapping, lattice, links):
    """Experimental local-normal W_in=0 on left/right/top; no oblique guarantee."""
    rho0, theta0, c0 = mapping.lattice.rho_ref_lu, mapping.theta_ref_lu, mapping.c0_lu
    for face, (axis, sign) in enumerate(((0, -1), (0, 1), (1, 1))):
        mask = links.outer_face == face
        if not np.any(mask):
            continue
        f, g = ((f_post[:, 0], g_post[:, 0]) if face == 0 else
                ((f_post[:, -1], g_post[:, -1]) if face == 1 else (f_post[-1], g_post[-1])))
        state = recover_macro(f, g, D=mapping.lattice.D, S=mapping.lattice.S, lattice=lattice)
        dp = state.p-rho0*theta0
        outgoing = dp+rho0*c0*sign*state.u[..., axis]
        entropy = state.rho-rho0-dp/c0**2
        rho_target = rho0+entropy+.5*outgoing/c0**2
        p_target = rho0*theta0+.5*outgoing
        u_target = state.u.copy()
        u_target[..., axis] = sign*.5*outgoing/(rho0*c0)
        fe, ge = equilibrium_fg(state.rho, state.u, state.theta, mapping.lattice.S, lattice)
        ft, gt = equilibrium_fg(rho_target, u_target, p_target/rho_target, mapping.lattice.S, lattice)
        ft, gt = ft+f-fe, gt+g-ge
        indices = _select(links.outer_in, mask)
        sample = links.outer_tangent[mask]
        sample_next, fraction = links.outer_tangent_next[mask], links.outer_fraction[mask]
        direction = links.outer_direction[mask]
        f_new[indices] = (1-fraction)*ft[sample, direction]+fraction*ft[sample_next, direction]
        g_new[indices] = (1-fraction)*gt[sample, direction]+fraction*gt[sample_next, direction]
