"""局部物性必须进入真实碰撞和热流读出，而非只修改映射元数据。"""
from pathlib import Path

import numpy as np
import pytest
import h5py

from core.collision_smrt import collide_fg
from core.macroscopic import central_energy_flux_lu, heat_flux_lu, recover_macro
from core.solver import GasSolver2D
from core.unit_mapping import create_unit_mapping, local_transport_coefficients
from config_io import load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs/gas_air_10k_d2q37_qmatched_local_transport.yaml"


def local_config(model="sutherland"):
    cfg = load_config(CONFIG)
    cfg["transport"] = {"mode": "local", "model": model,
                        "sutherland_mu_K": 111.0, "sutherland_k_K": 194.0}
    cfg["numerics"].update(nx=3, ny=2)
    return cfg


def prepared_state(model="sutherland"):
    s = GasSolver2D(local_config(model))
    rho = np.array([[.8, 1., 1.2], [1.2, .8, 1.]])
    temp_ratio = np.array([[.9, 1., 1.2], [1., 1.2, .9]])
    s.initialize_from_macro(rho, np.zeros(2), temp_ratio * s.mapping.theta_ref_lu)
    return s, rho, temp_ratio


def expected_tau(s, rho, tr, channel):
    m = s.mapping
    c = 111. if channel == "shear" else 194.
    factor = tr**1.5 * (m.physical.T0_K + c) / (m.physical.T0_K * tr + c)
    diffusivity = m.nu_lu if channel == "shear" else m.alpha_lu
    return .5 + diffusivity * factor / rho / (m.theta_ref_lu * tr)


def test_local_density_and_temperature_change_actual_stress_relaxation():
    s, rho, tr = prepared_state()
    cx, cy = s.lattice.c.T
    basis = cx * cy
    f = s.f + 1e-6 * s.lattice.w * basis
    fp, gp, diag = collide_fg(f, s.g, s.mapping, lattice=s.lattice, return_diagnostics=True)
    before = np.sum((f - s.f) * basis, axis=-1)
    after = np.sum((fp - s.f) * basis, axis=-1)
    np.testing.assert_allclose(after / before, 1 - 1 / expected_tau(s, rho, tr, "shear"), atol=2e-8)
    assert np.max(np.abs(diag.energy_residual)) < 1e-13
    assert fp.min() > 0 and gp.min() > 0


def test_local_heat_relaxation_and_readout_use_the_same_tau():
    s, rho, tr = prepared_state()
    g = s.g + 1e-6 * s.lattice.w * s.lattice.c[:, 0]
    q0 = central_energy_flux_lu(s.f, g, lattice=s.lattice)
    fp, gp = collide_fg(s.f, g, s.mapping, lattice=s.lattice)
    q1 = central_energy_flux_lu(fp, gp, lattice=s.lattice)
    tau = expected_tau(s, rho, tr, "thermal")
    np.testing.assert_allclose(q1[..., 0] / q0[..., 0], 1 - 1 / tau, atol=2e-8)
    # 离散时间的物理热流为碰撞前后原始中心能流的平均。
    q = heat_flux_lu(s.f, g, lattice=s.lattice, mapping=s.mapping)
    np.testing.assert_allclose(q, .5 * (q0 + q1), atol=1e-13)


def test_local_mode_rejects_unknown_material_law():
    with pytest.raises(ValueError, match="transport.*model"):
        create_unit_mapping(local_config("typo"))


def test_local_mode_rejects_inconsistent_heat_capacity():
    cfg = local_config()
    cfg["physical"]["cp_J_kgK"] *= 1.1
    with pytest.raises(ValueError, match="cp"):
        create_unit_mapping(cfg)


def test_moving_stressed_gas_preserves_heat_flux_relaxation_and_energy():
    s, rho, tr = prepared_state()
    u = np.array([.03, -.02])
    s.initialize_from_macro(rho, u, tr * s.mapping.theta_ref_lu)
    # 非平衡迹使 f/g 交换内能；该能量转移不能产生额外的随体热流。
    cx, cy = s.lattice.c.T
    f = s.f + 1e-5 * s.lattice.w * (cx * cx + cy * cy - 2 * s.lattice.theta_q)
    g = s.g + 1e-6 * s.lattice.w * cx
    before = recover_macro(f, g, lattice=s.lattice)
    coef = local_transport_coefficients(s.mapping, before.rho, before.theta)
    q0 = central_energy_flux_lu(f, g, lattice=s.lattice)
    fp, gp, diag = collide_fg(f, g, s.mapping, lattice=s.lattice, return_diagnostics=True)
    q1 = central_energy_flux_lu(fp, gp, lattice=s.lattice)
    np.testing.assert_allclose(q1, coef.heat_flux_retention[..., None] * q0, atol=3e-13)
    assert np.max(np.abs(diag.energy_residual)) < 1e-13


@pytest.mark.parametrize("rho,theta,error", [(0, .7, "density"), (-1, .7, "density"),
    (np.nan, .7, "density"), (1, 0, "temperature"), (1, np.inf, "temperature")])
def test_invalid_local_state_fails_without_clipping(rho, theta, error):
    m = create_unit_mapping(local_config())
    with pytest.raises(ValueError, match=error):
        local_transport_coefficients(m, rho, theta)


def test_constant_material_law_keeps_mu_and_k_independent_of_density():
    s, rho, tr = prepared_state("constant")
    a = local_transport_coefficients(s.mapping, rho, tr * s.mapping.theta_ref_lu)
    np.testing.assert_allclose(a.nu_lu * rho, s.mapping.nu_lu)
    np.testing.assert_allclose(a.alpha_lu * rho, s.mapping.alpha_lu)
    np.testing.assert_array_equal(a.tau22, .5)


def test_reference_state_recovers_reference_tau_and_nonzero_bulk_scales_locally():
    cfg = local_config()
    cfg["collision"].update(bulk_viscosity_policy="specified", nu_b_lu=.01)
    m = create_unit_mapping(cfg)
    a = local_transport_coefficients(m, 1., m.theta_ref_lu)
    assert a.tau21 == pytest.approx(m.tau21)
    assert a.tau22 == pytest.approx(m.tau22)
    assert a.tau32 == pytest.approx(m.tau32)
    b = local_transport_coefficients(m, 2., m.theta_ref_lu)
    assert b.tau22 - .5 == pytest.approx(.5 * (m.tau22 - .5))


@pytest.mark.parametrize("model", ["constant", "sutherland"])
def test_hdf5_export_preserves_fields_and_transport_metadata(tmp_path, model):
    cfg = local_config(model)
    s = GasSolver2D(cfg)
    s.initialize_from_macro(1., np.zeros(2), s.mapping.theta_ref_lu)
    s.step(2)
    path = tmp_path / f"{model}.h5"
    s.save_hdf5(str(path))
    with h5py.File(path, "r") as data:
        assert data.attrs["transport_mode"] == "local"
        assert data.attrs["transport_model"] == model
        assert data["metadata/unit_mapping"].attrs["tau_metadata_scope"] == "reference_state"
        np.testing.assert_array_equal(data["fields/f"][:], s.f)
        np.testing.assert_allclose(data["fields/q_lu"][:], s.get_heat_flux_lu())


@pytest.mark.parametrize("transport", [None, {}, {"model": "sutherland"}, {"mode": "frozen"}])
def test_retired_or_implicit_transport_cannot_run_as_current_physics(transport):
    cfg = local_config()
    if transport is None:
        cfg.pop("transport")
    else:
        cfg["transport"] = transport
    with pytest.raises(ValueError, match="transport.mode.*local"):
        GasSolver2D(cfg)


@pytest.mark.parametrize("key,value", [
    ("regularized_heat_flux_factor", -.87),
    ("regularized_heat_flux_factor_policy", "specified"),
    ("conductive_heat_flux_moment_factor", .0635448),
    ("conductive_heat_flux_moment_factor_policy", "specified"),
    ("conductive_heat_flux_galilean_correction_factor", .038),
    ("heat_flux_retention_policy", "specified"),
])
def test_retired_heat_flux_knobs_are_rejected_instead_of_silently_ignored(key, value):
    cfg = local_config()
    cfg["collision"][key] = value
    with pytest.raises(ValueError, match=key):
        GasSolver2D(cfg)
