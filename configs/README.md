# configs/

| 文件 | 用途 |
|---|---|
| [gas_air_10k_d2q37_qmatched_local_transport.yaml](gas_air_10k_d2q37_qmatched_local_transport.yaml) | 当前唯一气体基线：D2Q37 + 局部 Sutherland 输运；基础网格为周期体内夹具，算例层可展开为开放平面设置 |
| [phase4_kirchhoff_fixture.yaml](phase4_kirchhoff_fixture.yaml) | 独立 Helmholtz 制造解夹具，不是气体配置 |

算例参数放在 `Compute/<专题>/study.yaml`，不要为每个扫描点复制公共基线。
公共基线中的 `10k` 是沿用文件名，不会替算例设置驱动频率。

配置必须显式写 `transport.mode: local`；`model: sutherland` 按温度更新 μ、k。
`model: constant` 只固定 μ、k，仍由局部密度、温度计算 ν、α 和 τ，供物性对照使用。
热流系数由局部 τ32 推导，不接受手动固定系数或 Galilean 修正参数。

frozen 输运及 `gas_air_10k_d2q37_qmatched_analytic_stress.yaml` 已退役；历史结果用原提交复现。
缺少输运模式的旧配置同样报错，不能静默换成新模型。格子缺省为 D2Q37，显式 D2Q21 报错。

基线中的 `central_moment_closure`、`high_order_relaxation`、`trace_bulk_policy`、
`deviatoric_stress_policy`、`regularized_shear_xy_factor`、`regularized_shear_normal_factor`
记录唯一支持的碰撞行为，不是切换分支的旋钮。支持取值由 `core.unit_mapping` 校验。

底层 `config_io.load_config` 的 `include` 按顶层键覆盖；若覆盖 `physical` 等段，
应提供完整段。`study.yaml` 的 `defaults` / 单算例运行参数使用递归合并，两者不能混淆。
