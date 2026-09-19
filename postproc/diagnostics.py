"""Reusable complex-amplitude records, comparisons and cycle-window fits."""
import numpy as np
from postproc.multiharmonic_fit import fit_multiharmonic


def record(z):
    z = complex(z)
    return dict(re=z.real, im=z.imag, amplitude=abs(z), phase_deg=float(np.angle(z, deg=True)))


def compare(got, expected):
    return dict(amplitude_error_percent=100*(abs(got)/abs(expected)-1),
                phase_error_deg=float(np.angle(got/expected, deg=True)),
                complex_relative_error=float(abs(got/expected-1)))


def fit_window(t, x, frequency, start_cycle, end_cycle, *, harmonics=3, detrend_order=1):
    t, x = np.asarray(t), np.asarray(x)
    mask = (t*frequency >= start_cycle-1e-10) & (t*frequency < end_cycle-1e-10)
    fit = fit_multiharmonic(t[mask], x[mask], 2*np.pi*frequency,
                           n_harmonics=harmonics, detrend_order=detrend_order)
    return fit.harmonic(1), dict(n_samples=int(np.sum(mask)), residual_rms=fit.residual_rms,
        residual_over_fundamental=fit.residual_rms/abs(fit.harmonic(1)),
        h2_over_h1=fit.amplitude(2)/fit.amplitude(1) if harmonics >= 2 else 0.,
        condition_number=fit.condition_number)
