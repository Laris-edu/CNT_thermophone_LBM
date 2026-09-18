# 链路接口与适用范围

本文只写本仓代码已经建立的约定。物理验证状态与限制见 [README](../README.md)。

## 入口

```python
from coupling.conjugate import run_levelc_predictor_corrector
from coupling.film_ode import FilmOdeParams

result = run_levelc_predictor_corrector(
    solver=solver, params=FilmOdeParams(C_A_si=7e-4, T_ref_K=300, gas_flux_factor=2),
    drive=power_watt_per_m2, n_steps=40320, wall_bc="mass_neutral", grad_extrap="row1")
```

默认 `scheme="conservative"` 委托 [coupling/conservative.py](../coupling/conservative.py)；显式选择 `scheme="heun_picard1"` 时保留旧壁（`equilibrium_clamp`）与旧外推默认，不静默改变旧调用。

## 保守路径的硬约束（在入口处检查，不满足即报错）

- **封闭周期对称域**：`gas_flux_factor` 必须为 2（一张膜的两侧）；气体储能含湿壁节点，热带两侧合计对应一张膜的面积。
- `solver.ny >= 5`，`grad_extrap ∈ {row1, linear}`；`dt_si` 若给出必须等于 LBM 时间步。
- 不接受 row1 反馈、热流增益、松弛或其它壁的组合（旧 Heun 路径才允许这些诊断组合）。

## 逐步算法（每步一次碰撞-流步，壁在流步后联立）

1. `win[i+1] = 0.5·dt·(P[i]+P[i+1])`，输入功端点梯形。
2. 流步后用第 0 行重构前的布居读出湿壁行能量 `row_energy` 与该行密度 `rho_wall`，`Cw = cv·rho_wall·ΔE`（`cv=(D+S)/2`），这是膜温度联立方程里唯一与气体耦合的系数，不是拟合参数。
3. 解 `ΔT_s = (win − G·dt·(T_s−T_ref) + row_energy − Cw·T_s) / (C_A + Cw + 0.5·dt·G)`，即 `C_A(T_{n+1}−T_n) + ΔE_g + W_leak = W_in`，`T_wall = T_{n+1}`。
4. 用同一温度重建壁行（对称 mass-neutral 回调），记录该区间实际交换量 `exchange`。
5. 输出 `q_g_one_sided_si = exchange/(2·dt)`，即单侧热流；`q[0]=0` 为占位，`q[1:]` 属于相邻时间区间（`q_sample_kind="preceding_interval_mean"`），谐波拟合用区间中点、积分用 `sum(q[1:]·dt)`。

气体实际储能 `gas_energy_J_m2`、膜储能、输入功与漏热各自独立记录，审计比较的是它们的和，避免自洽但不守恒的假通过。无驱动平衡态有一个约 1e-14 J/m² 量级的归一化下限，只用以免把舍入级交换量当分母，不改变分布或残差。

## 状态与失效

- 不裁剪、不设经验增益、不用压力导数反馈；出现非有限值或非正膜温直接抛 `FloatingPointError`。
- 碰撞核要求平衡态逐布居非负（`core/equilibrium.py`）：超出可用温度/速度域会抛 `EquilibriumDomainError`，不会静默裁剪。静止可用域约 `θ/θ_q ∈ (0.84, 2.0]`，随马赫数收窄。
- 求解器是 CPU NumPy 向量化实现，`collide_fg` 占单步约 96%；未接入 GPU 或多核域分解。

## 探针与复声压

探头读数是**总压**，不是声压。周期声压评估需要先用 [postproc/multiharmonic_fit.py](../postproc/multiharmonic_fit.py) 分离均值、热漂移与目标谐波；`sample_interval` 必须覆盖整周期且逐步采样（旧的单音拟合器不处理趋势）。有限域周期压力响应不等于开放空气中的辐射声压。
