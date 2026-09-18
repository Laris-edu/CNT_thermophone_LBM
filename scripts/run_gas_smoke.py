"""新项目气体启动检查：小幅等压温度扰动、有限步推进与守恒检查。"""
from pathlib import Path
import argparse
import json

import numpy as np

from config_io import load_config
from core.solver import GasSolver2D

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml')
    parser.add_argument('--steps', type=int, default=64)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    solver = GasSolver2D(load_config(args.config))
    factor = 1.0 + 1e-5*np.sin(2*np.pi*np.arange(solver.nx)/solver.nx)
    solver.initialize_from_macro(solver.mapping.lattice.rho_ref_lu/factor, np.zeros(2), solver.mapping.theta_ref_lu*factor)
    before = solver.get_macro()
    mass0, energy0 = np.sum(before.rho), np.sum(before.E_tot)
    solver.step(args.steps)
    after = solver.get_macro()
    report = dict(scope='periodic_gas_startup_only',config=str(args.config.resolve()),steps=args.steps,
                  transport=solver.mapping.transport.mode,velocity_set=solver.mapping.lattice.velocity_set,
                  finite=bool(np.isfinite(solver.f).all() and np.isfinite(solver.g).all()),
                  mass_relative_error=float(abs(np.sum(after.rho)/mass0-1)),
                  energy_relative_error=float(abs(np.sum(after.E_tot)/energy0-1)),
                  temperature_evolution_max_lu=float(np.max(abs(after.theta-before.theta))))
    report['passed'] = bool(report['finite'] and report['mass_relative_error']<1e-11
                            and report['energy_relative_error']<1e-11
                            and report['temperature_evolution_max_lu']>0)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
