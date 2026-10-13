"""Optional MATLAB regression for the closed-curve acoustic handoff.

Run with RUN_KWAVE_TESTS=1. These are manufactured acoustic fields, not a
validation of the LBM thermal source or of a finite physical device.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.io import loadmat, savemat
from scipy.special import hankel2


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "Compute/finite_strip/kwave_handoff.m"
MATLAB = Path(os.environ.get("MATLAB_EXE", "E:/Matlab/bin/matlab.exe"))
TOOLBOX = Path(os.environ.get("KWAVE_TOOLBOX", str(ROOT / "k-wave-toolbox-version-1.4.1")))


def _field(x: np.ndarray, y: np.ndarray, *, source_x=0.001, source_y=-0.00075
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A displaced cylindrical monopole with a nonzero absolute phase."""
    k = 2 * np.pi * 20_000 / 347
    rx, ry = np.asarray(x) - source_x, np.asarray(y) - source_y
    radius = np.hypot(rx, ry)
    scale = 0.9 * np.exp(0.4j)
    p = scale * hankel2(0, k * radius)
    ur = -1j * scale * hankel2(1, k * radius) / (1.16 * 347)
    return p, ur * rx / radius, ur * ry / radius


def _input(dx: float, *, reverse_velocity: bool = False) -> dict:
    theta = (np.arange(256) + 0.5) * (2 * np.pi / 256)
    x, y = 0.008 * np.cos(theta), 0.008 * np.sin(theta)
    p, ux, uy = _field(x, y)
    un = ux * np.cos(theta) + uy * np.sin(theta)
    return {
        "x_m": x,
        "y_m": y,
        "normal_x": np.cos(theta),
        "normal_y": np.sin(theta),
        "ds_m": np.full(theta.size, 2 * np.pi * 0.008 / theta.size),
        "p_hat_Pa": p,
        "un_hat_m_s": -un if reverse_velocity else un,
        "rho0_kg_m3": 1.16,
        "c0_m_s": 347.0,
        "frequency_Hz": 20_000.0,
        "dx_m": dx,
        "domain_half_width_m": 0.048,
        "domain_half_height_m": 0.048,
        "evaluation_cycles": 10.25,
        "ramp_cycles": 3.0,
        "record_cycles": 2.0,
        "cfl": 0.2,
        "pml_size": 20,
        "observer_x_m": np.array([0, 0.018, 0, 0, 0, 0.003, -0.003]),
        "observer_y_m": np.array([0.015, 0.024, 0.045, -0.045, 0, 0, 0.003]),
        "toolbox_path": str(TOOLBOX),
        "source_validation_passed": True,
    }


@pytest.fixture(scope="module")
def handoff_results(tmp_path_factory):
    if os.environ.get("RUN_KWAVE_TESTS") != "1":
        pytest.skip("set RUN_KWAVE_TESTS=1 to run the optional MATLAB handoff regression")
    if not MATLAB.is_file() or not TOOLBOX.is_dir():
        pytest.skip("MATLAB executable and k-Wave toolbox are required")
    assert ADAPTER.is_file(), "The closed-curve k-Wave adapter has not been implemented"
    work = tmp_path_factory.mktemp("strip_handoff")
    cases = {"coarse": _input(0.00075), "fine": _input(0.0005),
             "wrong_velocity": _input(0.0005, reverse_velocity=True)}
    statements = ["addpath('" + ADAPTER.parent.as_posix().replace("'", "''") + "')"]
    for name, config in cases.items():
        incoming, outgoing = work / f"{name}_in.mat", work / f"{name}_out.mat"
        savemat(incoming, config)
        quote = lambda path: "'" + path.as_posix().replace("'", "''") + "'"
        statements.append(f"kwave_handoff({quote(incoming)}, {quote(outgoing)})")
    run = subprocess.run([str(MATLAB), "-batch", ";".join(statements)],
                         cwd=ROOT, capture_output=True, timeout=300)
    assert run.returncode == 0, (run.stdout + run.stderr).decode(errors="replace")
    return {name: loadmat(work / f"{name}_out.mat", squeeze_me=True)
            for name in cases}


def _errors(result: dict) -> tuple[float, float, float]:
    prediction = np.asarray(result["observer_p_hat_Pa"])
    reference = _field(result["observer_x_m"], result["observer_y_m"])[0]
    ratio = prediction[:4] / reference[:4]
    amplitude = float(np.max(np.abs(np.abs(ratio) - 1)))
    phase = float(np.max(np.abs(np.angle(ratio, deg=True))))
    interior = float(np.max(np.abs(prediction[4:])) / np.max(np.abs(reference[:4])))
    return amplitude, phase, interior


def test_closed_curve_reproduces_exterior_and_cancels_interior(handoff_results):
    coarse = _errors(handoff_results["coarse"])
    fine = _errors(handoff_results["fine"])
    print({"coarse": coarse, "fine": fine})
    assert fine[0] < 0.01
    assert fine[1] < 1.0
    assert fine[2] < 0.01
    assert fine[0] < coarse[0], "Spatial refinement must improve amplitude accuracy"
    assert fine[2] < coarse[2], "Spatial refinement must improve interior cancellation"


def test_velocity_channel_has_discriminating_power(handoff_results):
    amplitude, phase, interior = _errors(handoff_results["wrong_velocity"])
    assert amplitude > 0.5 or phase > 30
    assert interior > 0.5


def test_diagnostic_gate_preserves_source_status_and_structural_checks(tmp_path):
    """Short opt-in gate test; its tiny manufactured field tests no accuracy claim."""
    if os.environ.get("RUN_KWAVE_GATE_TESTS") != "1":
        pytest.skip("set RUN_KWAVE_GATE_TESTS=1 for the short MATLAB gate regression")
    if not MATLAB.is_file() or not TOOLBOX.is_dir():
        pytest.skip("MATLAB executable and k-Wave toolbox are required")
    base = _input(0.002)
    base.update(domain_half_width_m=0.012, domain_half_height_m=0.012,
                observer_x_m=np.array([0.0]), observer_y_m=np.array([0.010]),
                evaluation_cycles=3.25, ramp_cycles=1.0, record_cycles=1.0,
                cfl=0.3, pml_size=4, source_validation_passed=False)
    cases = {
        "default_rejected": dict(base),
        "explicit_rejected": dict(base, diagnostic_mode=0),
        "diagnostic_reaches_dependency": dict(base, diagnostic_mode=1),
        "missing_status": dict(base, diagnostic_mode=1),
        "invalid_status": dict(base, source_validation_passed=2, diagnostic_mode=1),
        "invalid_mode": dict(base, diagnostic_mode=2),
        "nonfinite_source": dict(base, diagnostic_mode=1),
        "open_curve": dict(base, diagnostic_mode=1),
        "diagnostic_output": dict(base, diagnostic_mode=1),
        "validated_output": dict(base, source_validation_passed=True),
    }
    cases["diagnostic_reaches_dependency"].pop("toolbox_path")
    cases["missing_status"].pop("source_validation_passed")
    cases["nonfinite_source"]["p_hat_Pa"] = np.full(256, np.nan)
    cases["open_curve"]["normal_x"] = np.ones(256)
    cases["open_curve"]["normal_y"] = np.zeros(256)
    quote = lambda path: "'" + path.as_posix().replace("'", "''") + "'"
    statements = [f"addpath({quote(ADAPTER.parent)})"]
    for name, config in cases.items():
        incoming, outgoing = tmp_path / f"{name}_in.mat", tmp_path / f"{name}_out.mat"
        savemat(incoming, config)
        error_path = tmp_path / f"{name}_error.mat"
        statements.append(
            f"failure='';try,kwave_handoff({quote(incoming)},{quote(outgoing)});"
            f"catch exception,failure=exception.message;end;"
            f"save({quote(error_path)},'failure','-v7')")
    options = {}
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        options.update(startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
    process = subprocess.Popen([str(MATLAB), "-singleCompThread", "-batch", ";".join(statements)],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
    try:
        stdout, stderr = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            process.kill()
        process.communicate()
        pytest.fail("Short MATLAB gate regression exceeded 90 seconds")
    log = (stdout + stderr).decode(errors="replace")
    assert process.returncode == 0, log
    errors = {name: str(loadmat(tmp_path / f"{name}_error.mat", squeeze_me=True)["failure"])
              for name in cases}
    assert "toolbox_path" in errors["diagnostic_reaches_dependency"], errors
    for name in ("default_rejected", "explicit_rejected"):
        assert "upstream source must pass" in errors[name], errors
    for name in ("missing_status", "invalid_status"):
        assert "source_validation_passed" in errors[name], errors
    assert "diagnostic_mode" in errors["invalid_mode"], errors
    assert "finite" in errors["nonfinite_source"], errors
    assert "normals do not close" in errors["open_curve"], errors
    assert "DIAGNOSTIC" in log and "source_validation_passed=0" in log, log
    for name, passed, diagnostic, bypassed in (
            ("diagnostic_output", False, True, True),
            ("validated_output", True, False, False)):
        result = loadmat(tmp_path / f"{name}_out.mat", squeeze_me=True)
        assert bool(result["source_validation_passed"]) is passed
        assert bool(result["diagnostic_mode"]) is diagnostic
        assert bool(result["source_validation_bypassed"]) is bypassed
        assert np.isfinite(result["observer_p_hat_Pa"]).all()
    for name in cases.keys() - {"diagnostic_output", "validated_output"}:
        assert not (tmp_path / f"{name}_out.mat").exists()


def run_thin_rectangle_fixture(output_directory: Path, *, dx_m: float,
                               pml_size: int = 20, domain_half_height_m: float = 0.05,
                               evaluation_cycles: float = 10.25) -> dict:
    """Run one independently manufactured thin-curve check, never LBM data.

    This callable writes diagnostic MAT files only into a new directory.
    The physical curve is x=+/-0.7 mm, y=+/-0.2 mm; its source is (0.1 mm, 0).
    It is intentionally separate from the large-circle regression.
    """
    output_directory.mkdir(parents=True, exist_ok=False)
    half_width, half_height, step = 0.0007, 0.0002, 0.0000125
    x = np.linspace(-half_width, half_width, 113)
    y = np.linspace(-half_height, half_height, 33)
    xm, ym = (x[1:] + x[:-1]) / 2, (y[1:] + y[:-1]) / 2
    sx = np.concatenate((xm, np.full(ym.size, half_width), xm[::-1],
                         np.full(ym.size, -half_width)))
    sy = np.concatenate((np.full(xm.size, -half_height), ym,
                         np.full(xm.size, half_height), ym[::-1]))
    nx = np.concatenate((np.zeros(xm.size), np.ones(ym.size),
                         np.zeros(xm.size), -np.ones(ym.size)))
    ny = np.concatenate((-np.ones(xm.size), np.zeros(ym.size),
                         np.ones(xm.size), np.zeros(ym.size)))
    pressure, ux, uy = _field(sx, sy, source_x=0.0001, source_y=0)
    cfg = _input(dx_m)
    cfg.update(x_m=sx, y_m=sy, normal_x=nx, normal_y=ny,
               ds_m=np.full(sx.size, step), p_hat_Pa=pressure,
               un_hat_m_s=ux * nx + uy * ny,
               domain_half_width_m=0.004, domain_half_height_m=domain_half_height_m,
               cfl=0.3, pml_size=pml_size, evaluation_cycles=evaluation_cycles,
               observer_x_m=np.array([0, 0, 0, 0, 0, -0.0004, 0.0004]),
               observer_y_m=np.array([0.002, 0.01, 0.045, -0.045, 0, 0, 0]))
    incoming, outgoing = output_directory / "input.mat", output_directory / "output.mat"
    savemat(incoming, cfg)
    quote = lambda path: "'" + path.as_posix().replace("'", "''") + "'"
    command = (f"addpath({quote(ADAPTER.parent)});"
               f"kwave_handoff({quote(incoming)}, {quote(outgoing)})")
    process = subprocess.Popen([str(MATLAB), "-batch", command], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        # MATLAB on Windows has a launcher plus a computation child. Kill only
        # this test's process tree, not other MATLAB sessions on the computer.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        (output_directory / "matlab.log").write_bytes(stdout + stderr)
        raise TimeoutError("Thin-curve diagnostic exceeded its 600 s budget") from None
    (output_directory / "matlab.log").write_bytes(stdout + stderr)
    if process.returncode:
        raise RuntimeError((stdout + stderr).decode(errors="replace"))
    result = loadmat(outgoing, squeeze_me=True)
    reference = _field(result["observer_x_m"], result["observer_y_m"],
                       source_x=0.0001, source_y=0)[0]
    prediction = result["observer_p_hat_Pa"]
    ratio = prediction[:4] / reference[:4]
    metrics = {
        "dx_m": dx_m, "pml_size": pml_size,
        "domain_half_height_m": domain_half_height_m,
        "evaluation_cycles": evaluation_cycles,
        "amplitude_error_max": float(np.max(np.abs(np.abs(ratio) - 1))),
        "phase_error_deg_max": float(np.max(np.abs(np.angle(ratio, deg=True)))),
        "interior_relative_max": float(np.max(np.abs(prediction[4:]))
                                       / np.max(np.abs(reference[:4]))),
        "amplitude_error_45mm": float(abs(abs(ratio[2]) - 1)),
        "phase_error_45mm_deg": float(abs(np.angle(ratio[2], deg=True))),
        "elapsed_s": float(result["elapsed_s"]),
    }
    metrics["validation_passed"] = bool(
        metrics["amplitude_error_max"] < 0.01
        and metrics["phase_error_deg_max"] < 1.0
        and metrics["interior_relative_max"] < 0.01
    )
    return metrics


def test_thin_rectangle_requires_resolved_geometry(tmp_path):
    """Opt-in thin-curve regression; the fine MATLAB run takes several minutes."""
    if os.environ.get("RUN_KWAVE_THIN_TESTS") != "1":
        pytest.skip("set RUN_KWAVE_THIN_TESTS=1 for the slower thin-curve regression")
    if not MATLAB.is_file() or not TOOLBOX.is_dir():
        pytest.skip("MATLAB executable and k-Wave toolbox are required")
    coarse = run_thin_rectangle_fixture(tmp_path / "coarse", dx_m=0.0001)
    fine = run_thin_rectangle_fixture(tmp_path / "fine", dx_m=0.00005, pml_size=32,
                                      domain_half_height_m=0.049, evaluation_cycles=8.25)
    assert coarse["interior_relative_max"] > 0.01  # Detect an unresolved thin source layer.
    assert fine["validation_passed"]
    assert fine["interior_relative_max"] < coarse["interior_relative_max"]
