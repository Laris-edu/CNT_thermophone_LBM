"""实际演化验证局部物性，不能用系数计算自身作为唯一证据。"""
from verification.local_transport_measurement import measure_uniform, measure_nonuniform, CONFIG
from config_io import load_config
from verification.acoustic_wave_measurement import AcousticWaveSettings, measure_acoustic_wave_direction


def test_hot_dilute_moving_gas_recovers_material_transport():
    row = measure_uniform(T=360., density_ratio=.8, speed=.03)
    assert abs(row["nu_ratio"] - 1) < .01, row
    assert abs(row["alpha_ratio"] - 1) < .01, row
    assert abs(complex(row["q_ratio_real"], row["q_ratio_imag"]) - 1) < .01, row
    assert row["mass_relative_error"] < 1e-12 and row["energy_relative_error"] < 1e-12
    assert row["min_f"] > 0 and row["min_g"] > 0


def test_nonuniform_finite_temperature_matches_nsf_with_refinement():
    coarse = measure_nonuniform()
    fine = measure_nonuniform(refine=2)
    # 检查相对于真实温度变化量的误差，避免巨大背景温度掩盖错误。
    assert fine["temperature_evolution_error"] < .01, fine
    assert fine["temperature_evolution_error"] < .4 * coarse["temperature_evolution_error"]
    assert fine["heat_flux_relative_error"] < .003, fine
    assert fine["heat_flux_relative_error"] < .4 * coarse["heat_flux_relative_error"]
    assert fine["mass_relative_error"] < 1e-12 and fine["energy_relative_error"] < 1e-12
    assert fine["min_f"] > 0 and fine["min_g"] > 0


def test_reference_acoustic_attenuation_is_preserved():
    row = measure_acoustic_wave_direction(load_config(CONFIG), "x",
        AcousticWaveSettings(nx=128, ny=4, steps=300, fit_start=40, sample_interval=1, amplitude=1e-6))
    assert abs(row["acoustic_attenuation_measured_lu"] / row["acoustic_attenuation_reference_lu"] - 1) < .03, row
    assert row["sound_speed_relative_error"] < .002, row
