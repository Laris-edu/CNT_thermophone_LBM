# 迁移溯源

2026-09-17 从 `E:\JFM2024` 迁出的核心链路仓。逐文件来源、SHA256 与删除清单见 [migration_manifest.json](migration_manifest.json)；本页记录方法与证据。

## 方法

1. 以链路入口（`coupling.conjugate`、`coupling.conservative`、`coupling.drive`、`coupling.energy_audit`、`coupling.film_ode`、`phase3_interfaces.heat_flux_extraction`、`phase3_interfaces.wall_state_contract`、`postproc.multiharmonic_fit`、`boundary.wall_thermal_mass_neutral`、`boundary.wall_thermal_grad`、`config_io`、`reference.constants`、`reference.continuum_1d_freq`）做全仓 import 闭包（含相对导入），得到 31 个模块；闭包外模块不被任何保留文件引用，可直接不带。
2. 保留文件机械复制；`configs/` 下两个基线 YAML 与旧仓**逐字节相同**（它们是已验收算例的输入记录）。
3. 只对被判定为死代码或已退役路线的部分做裁剪，裁剪前逐符号核对"是否被保留文件引用"。
4. 两个测试文件按新仓内容同步：D2Q21 断言改为"退役速度集立即报错"；删除已不存在的解析应变率分支的 xfail 用例。

## 等价性证据

在旧仓与新仓各跑同一算例（local 基线与 analytic_stress 两组配置，nx=64/ny=8，200 步），状态数组 SHA256 与宏观总量、松弛时间**逐位一致**：

| 配置 | f | g |
|---|---|---|
| local 基线 | `57e1dfad5748bd89` | `5f51c9c90ac96a83` |
| analytic_stress (frozen) | `13df47b43d57f511` | `1236e0f6f4c200f7` |

两仓结果相同，说明裁剪没有改变本项目实际运行路径上的任何数值。

## 回归

| 仓 | 结果 |
|---|---|
| 旧仓 `E:\JFM2024` | 93 passed, 1 xfailed |
| 本仓 | 92 passed |

差 2 项：`test_phase2_postprocess_modal_fit.py` 随 `modal_fit` 移除；解析应变率分支的 xfail 用例随该分支删除。`scripts.run_gas_smoke` 在本仓通过（质量/能量相对误差 ≤ 2.2e-16）。

## 删除清单（17 项，完整哈希见 manifest）

- **退役路线**：`core/lattice_d2q21.py`（四阶平衡态需八阶求积）、`core/retired_spectral.py`（DC/A/H 保护，改为在 `unit_mapping` 里显式拒绝旧取值）、`core/polyatomic_fg.py`（全仓无调用）、`core/hermite.py`（只保留被使用的 `monomial_exponents`，已内联）。
- **未认证边界**：`boundary/open_cbc.py`、`boundary/open_sponge.py`、`boundary/wall_neumann.py`、`boundary/wall_mass_audit.py`。
- **旧后处理/交接**：`phase3_interfaces/modal_fit.py`、`probe_sampling.py`、`run_hdf5.py`、`postproc/harmonic_fit.py`。
- **链路外参照与脚本**：`reference/continuum_1d_time.py`、`reference/nonlinear_nsf_1d.py`、`scripts/check_environment.py`、`scripts/verify_local_transport.py`、`verification/test_phase2_postprocess_modal_fit.py`。

**未迁入文件内被删除的分支**（裁剪量：删除 1912 行、新增 180 行）：

| 文件 | 删除内容 |
|---|---|
| `core/collision_smrt.py` | 解析应变率偏应力分支（旧仓审查实测其符号与实测口径相反、等效剪切黏性为负）、四阶闭合、`ghost_orthogonal_local*` 迹策略族、`acoustic_simplified_collision` 开关 |
| `core/unit_mapping.py` | 上述策略的曲线机构与常数、DC/A/H 与 `seam_aware` 旋钮、`alpha_repair_scale` 旁路、三个无调用的配置工厂（`physical_timestep_config`、`d2q37_physical_timestep_config`、`quadrature_matched_config`） |
| `core/solver.py` | 谱投影机构、压力记忆迹通道、退役配置保护与随之而来的元数据字段 |
| `core/macroscopic.py` | 退役保护调用与三个无调用工具 |
| `core/equilibrium.py` | 无调用的 Gaussian 矩目标工具 |
| `core/lattice.py` | D2Q21 注册与 `lattice_family` |

被删除的解析应变率分支缺陷的完整证据（剪切波测量、逐格符号、守恒残差、平衡态可用域）在旧仓 `docs/CORE_REVIEW_20260917.md` 与 `results/core_review_20260917/`。

## 行为差异（相对旧仓默认路径）

除上表删除项外，只有以下三处有意差异（前两项改变行为，第三项只改注释）：

1. `lattice.velocity_set` 缺省不再落到 D2Q21，而是 D2Q37；显式写 `D2Q21` 会立即报错（旧仓是构造成功、第一个时间步才失败）。
2. 旧配置若仍写着已删除的旋钮且取值与本内核支持的行为不同，`create_unit_mapping` 立即报错，避免旧配置被静默按新内核重跑。当前两个基线 YAML 的取值都落在支持范围内，因此配置未被修改。

3. `boundary/wall_thermal_mass_neutral.py` 与 `postproc/multiharmonic_fit.py` 各改了一句 docstring：原引用指向未迁入的模块（`open_cbc`、`nonlinear_nsf_1d`）。无代码行为变化，manifest 里标记为 `edited`。

`solver.save_hdf5` 的 dataset 键名、分组与 dtype 未变；`collision` 组的属性集按裁剪后的字段精简（删除了已退役策略的字段，改为记录字面量 `trace_bulk_policy="tau22"`），因此**新旧 HDF5 的属性集合不完全相同**，读取旧文件时不要假设属性齐全。`h5py` 改为在 `save_hdf5` 内部导入，`import core` 不再强依赖它。
