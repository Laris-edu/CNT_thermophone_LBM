"""One-step interface diagnostics using the public collision and open solvers.

The affine conductive-moment initialization is a local consistency probe, not a
boundary-value validation. Optional thermal-step replays use the public runner
and its four-file contract. Diagnostic explosion overrides are process-local,
preserve each donor mean, and never modify production source or saved cases.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np

from core import two_level_planar_solver as interface
from core.equilibrium import equilibrium_fg
from core.macroscopic import heat_flux_lu, recover_macro
from core.planar_open_solver import PlanarOpenSolver
from simulation import run_case
from simulation.multigrid import collect_synchronized
from simulation.provenance import capture_source
from simulation.study import ROOT, load_study


HERE = Path(__file__).resolve().parent
LINEAR = interface._linear_explode
COLLIDE = interface.collide_fg


def heat(f, g, mapping, lattice):
    return heat_flux_lu(f, g, mapping=mapping, lattice=lattice)[:, 0, 1]*mapping.heat_flux_scale


def split_probe(solver, mode):
    """Advance one actual synchronized step; return both fine endpoint moments."""
    donor_eq, collision_inputs = [], []
    split_index = 0

    def collision(f, g, mapping, **kwargs):
        collision_inputs.append((f.copy(), g.copy(), mapping))
        out = COLLIDE(f, g, mapping, **kwargs)
        if len(collision_inputs) == 1:
            macro = recover_macro(*out, D=mapping.lattice.D, S=mapping.lattice.S,
                                  lattice=solver.lattice)
            eq = equilibrium_fg(macro.rho, macro.u, macro.theta, mapping.lattice.S, solver.lattice)
            for direction, cy in enumerate(solver.lattice.c[:, 1]):
                if cy < 0:
                    donor_eq.extend((eq[0][:, :, direction], eq[1][:, :, direction]))
        return out

    def explode(values, count):
        nonlocal split_index
        eq = donor_eq[split_index]
        split_index += 1
        repeated = np.repeat(values[:count], 2, axis=0)
        if mode == 'repeat':
            return repeated
        if mode == 'linear':
            return LINEAR(values, count)
        field = eq if mode == 'equilibrium_slope' else values-eq
        return repeated+LINEAR(field, count)-np.repeat(field[:count], 2, axis=0)

    before = solver.get_heat_flux_lu()[:, 0, 1]*solver.mapping.heat_flux_scale
    with patch.object(interface, 'collide_fg', collision), patch.object(interface, '_linear_explode', explode):
        audit = solver.step_temperature(300.1)
    f, g, mapping = collision_inputs[2]
    half = heat(f, g, mapping, solver.lattice)
    final = solver.get_heat_flux_lu()[:, 0, 1]*solver.mapping.heat_flux_scale
    rows = np.arange(solver.nf-6, solver.nf+4)
    return dict(mode=mode, rows=rows.tolist(), z_m=solver.z_m[rows].tolist(),
                initial_q_W_m2=before[rows].tolist(),
                first_fine_endpoint_q_W_m2=half[-6:].tolist(),
                synchronized_q_W_m2=final[rows].tolist(),
                energy_residual_J_m2=audit.energy_residual_J_m2,
                interface_sum_SI=solver.last_interface_audit.as_array()[2].tolist(),
                min_f=float(solver.f.min()), min_g=float(solver.g.min()))


def variants(solver, uniform):
    checkpoint = solver.checkpoint()
    reference_initial = uniform.get_heat_flux_lu()[:, 0, 1]*uniform.mapping.heat_flux_scale
    uniform.step_temperature(300.1)
    half = uniform.get_heat_flux_lu()[:, 0, 1]*uniform.mapping.heat_flux_scale
    uniform.step_temperature(300.1)
    final = uniform.get_heat_flux_lu()[:, 0, 1]*uniform.mapping.heat_flux_scale
    rows = np.arange(solver.nf-6, solver.nf+4)
    sample = lambda q: np.interp(solver.z_m[rows], uniform.z_m, q).tolist()
    results = []
    for mode in ('repeat', 'equilibrium_slope', 'nonequilibrium_slope', 'linear'):
        solver.restore(checkpoint)
        results.append(split_probe(solver, mode))
    solver.restore(checkpoint)
    return dict(reference_initial_q_W_m2=sample(reference_initial),
                reference_first_endpoint_q_W_m2=half[solver.nf-6:solver.nf].tolist(),
                reference_synchronized_q_W_m2=sample(final), variants=results)


def initialize_affine(solver, gradient_K_m, position_m):
    """Initialize the leading-order steady conductive moment in an affine field.

    Linearize the existing equilibrium at constant pressure and zero velocity.
    Set f^neq=-tau32*dx*cy*d_y(feq), and likewise g. This is controlled
    initialization only; no analytic state is injected into later evolution.
    This is not an exact all-population CE state: unresolved kinetic components
    need not follow tau32. The uniform-grid heat drift is measured separately.
    """
    m = solver.mapping
    T0 = m.physical.T0_K

    def eq(T):
        rho = np.full((1, 1), m.lattice.rho_ref_lu*T0/T)
        return equilibrium_fg(rho, np.zeros((1, 1, 2)), np.full((1, 1), T/m.temperature_scale),
                              m.lattice.S, solver.lattice)

    base, plus, minus = eq(T0), eq(T0+.001), eq(T0-.001)
    tau = np.full((solver.ny, 1, 1), m.tau32)
    width = np.full_like(tau, m.lattice.dx_m)
    if hasattr(solver, 'nf'):
        tau[solver.nf:] = solver.coarse_mapping.tau32
        width[solver.nf:] = solver.coarse_mapping.lattice.dx_m
    distance = (solver.z_m-position_m)[:, None, None]
    for name, zero, high, low in zip(('f', 'g'), base, plus, minus):
        derivative = (high-low)/.002
        setattr(solver, name, zero+gradient_K_m*distance*derivative
                -tau*width*gradient_K_m*solver.lattice.c[:, 1]*derivative)


def affine_probes(config):
    config = deepcopy(config)
    config['numerics']['ny'] = 160
    uniform_config = deepcopy(config)
    uniform_config.pop('grid')
    output = []
    for gradient in (-1000., -100., 100.):
        solver = interface.TwoLevelPlanarOpenSolver(config)
        uniform = PlanarOpenSolver(uniform_config)
        for target in (solver, uniform):
            initialize_affine(target, gradient, config['grid']['interface_m'])
        output.append(dict(gradient_K_m=gradient,
                           intended_Fourier_q_W_m2=-solver.mapping.physical.kg_W_mK*gradient,
                           **variants(solver, uniform)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-batch', required=True, type=Path)
    parser.add_argument('--replay-step', action='store_true')
    args = parser.parse_args()
    batch = args.output_batch.resolve()
    if batch.exists():
        raise FileExistsError(batch)
    configs = {c['case']['name']: c for c in load_study(HERE/'study.yaml')}
    source = capture_source(ROOT)
    script = dict(path=str(Path(__file__).resolve()),
                  sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  git_status=subprocess.check_output(
                      ['git', '-C', str(ROOT), 'status', '--porcelain', '--',
                       Path(__file__).resolve().relative_to(ROOT).as_posix()],
                      text=True, encoding='utf-8').strip())
    for name in ('step_uniform', 'step_two_level') if args.replay_step else ():
        c = configs[name]
        print(json.dumps(dict(case=name, fine_steps=c['numerics']['steps'],
                              output=str(batch/'raw'/name))), flush=True)
    batch.mkdir(parents=True)
    result = dict(scope='Local one-step consistency and optional actual thermal-step replay',
                  source=source, diagnostic_script=script,
                  affine=affine_probes(configs['step_two_level']),
                  limitations=[
                      'Affine field matches leading conductive moments, not an exact full-population CE state or boundary solution.',
                      'Only interface-near rows are inspected, outside two-step wall/outlet reach.',
                      'Overrides affect only the explosion spatial slope; they are not proposed solvers.',
                      'One-step decomposition is about a shared checkpoint, not an accumulated-error decomposition.',
                      'No temporal interpolation or non-equilibrium rescaling is implemented.'])
    if args.replay_step:
        solvers = {}

        def collect(solver, output, run):
            data = collect_synchronized(solver, output, run)
            solvers[solver.config['case']['name']] = solver
            return data

        with patch('simulation.multigrid.collect_synchronized', collect):
            for name in ('step_uniform', 'step_two_level'):
                run_case(configs[name], batch/'raw'/name,
                         provenance=dict(**source, diagnostic_script=script))
        result['after_5us_checkpoint'] = variants(solvers['step_two_level'], solvers['step_uniform'])
    (batch/'final').mkdir()
    output = batch/'final/interface_diagnosis.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
