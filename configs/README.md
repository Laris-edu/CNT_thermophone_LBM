# configs/

| 文件 | 用途 |
|---|---|
| [gas_air_10k_d2q37_qmatched_local_transport.yaml](gas_air_10k_d2q37_qmatched_local_transport.yaml) | 当前基线：D2Q37 + 局部 Sutherland 输运；nx=128/ny=4 周期体内夹具 |
| [gas_air_10k_d2q37_qmatched_analytic_stress.yaml](gas_air_10k_d2q37_qmatched_analytic_stress.yaml) | frozen 小振幅对照：固定物性、显式热流系数与 Galilean 修正项 |
| [phase4_kirchhoff_fixture.yaml](phase4_kirchhoff_fixture.yaml) | 独立 Helmholtz 制造解夹具 |

两个基线 YAML 与旧仓逐字节相同（哈希见 [docs/migration_manifest.json](../docs/migration_manifest.json)）；它们记录的是已验收算例的输入，不要为迁移而改写。

**以下键在当前内核里已不生效**，仅作为历史记录保留在文件中：`model`、`use_binomial_transform`、`central_moment_closure`、`high_order_relaxation`、`trace_bulk_policy`、`deviatoric_stress_policy`、`regularized_shear_xy_factor`、`regularized_shear_normal_factor`、`heat_flux_retention_policy`、`heat_flux_retention_curve`。它们当前的值与本内核唯一支持的行为一致，因此配置仍被接受；一旦改成别的取值，`create_unit_mapping` 会报错而不是静默降级（见 `core/unit_mapping.py` 的 `_RETIRED_COLLISION_VALUES`）。

配置缺 `lattice` 段不再静默退化：`velocity_set` 缺省按 D2Q37 处理，显式写 `D2Q21` 会立即报错。
