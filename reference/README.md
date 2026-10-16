# reference/

独立参照与常数。参照只用于对照，不构成 LBM 器件认证。

| 文件 | 用途 |
|---|---|
| [continuum_1d_freq.py](continuum_1d_freq.py) | 一维半解析热声参照（`solve_level_C_frequency`），组合热导纳与解析声压剖面；不是独立离散的完整 NSF 求解器 |
| [lim2013.py](lim2013.py) | `solve_lim2013_frequency` 统一求解 Lim 近场复声压、远场 RMS、精确平面解及 Eq. (24) 声分支；独立解析参照 |
| [analytical_models.py](analytical_models.py) | 解析热声模型（含指数热模态声压源项）；薄热层、无黏近似 |
| [thermal_admittance.py](thermal_admittance.py) | 半空间热导纳 |
| [film_ode.py](film_ode.py) | 参考膜 ODE（对照用，与 `coupling/film_ode.py` 的耦合实现不同） |
| [constants.py](constants.py) | 默认物理常数与热尺度 |
| [result_schema.py](result_schema.py) | 参照结果字段约定 |

## Lim 2013 参数口径（改错任一处结果差一个因子）

| Lim 论文 | 本仓 | 关系 |
|---|---|---|
| `c_s` 单位面积热容 | `FilmOdeParams.C_A_si` | 1:1 |
| `beta0` 单侧热损 W/(m²K) | `linear_leak_conductance_si` | **本仓值 = 2β₀**（双侧合计） |
| `P_in/s`（投影面积口径） | 驱动 W/m² | 论文正文按 `P_in/(2s)` 写；本仓是 2 倍 |
| Eq. (16) 分母 | `level_c_closed_form` 分母 | ×2 后逐项相同 |
| `M = sqrt(jω/α)`（Eq. 18） | `thermal_admittance_halfspace` | 相同 |

### 公共频域求解入口

从项目根目录导入，无需启动 LBM 或写出文件。输入 `f_hz` 是**声频**，
`drive_areal_hat` 是投影面积口径的交流复面功率（W/m²），不包含平均功率。
论文的总功率 `P_in*(1-cos ωt)` 对应 `drive_areal_hat=-P_in/s`；任意谐波驱动按原复幅传入。
`area_m2` 为单侧膜面积，`c_s_J_m2K` 为膜本体面积热容，`beta0_per_side_W_m2K` 为单侧热损。
频率、面积、面积热容、观测距离须有限且为正，热损可为零，复驱动须有限。

```python
from reference.lim2013 import areal_drive_from_power, solve_lim2013_frequency

result = solve_lim2013_frequency(
    f_hz=1.0e4,
    drive_areal_hat=-areal_drive_from_power(0.85, 9.0e-4),
    c_s_J_m2K=1.0e-3,
    area_m2=9.0e-4,
    x_m=0.115,
    beta0_per_side_W_m2K=16.0,
)
print(result.rayleigh_distance_m, result.in_near_field)
print(result.p_hat_near_approximate_Pa, result.p_rms_near_approximate_Pa)
print(result.p_rms_far_Pa)
```

返回 `LimFrequencyResult`，同一频率和距离下各公式**分别**求值：

| 输出字段 | 定义与适用条件 |
|---|---|
| `T_s_hat_exact_K`、`q_one_sided_hat_exact_W_m2` | Eqs. (13)/(16) 的膜温与向外单侧导热热流 |
| `T_s_hat_approximate_K`、`q_one_sided_hat_approximate_W_m2` | 用 Eq. (18) 热导纳得到的膜温与单侧热流 |
| `p_hat_planar_exact_Pa`、`p_rms_planar_exact_Pa` | Eq. (9) 完整精确平面解，含热分支和声分支；不是有限膜球面场 |
| `p_hat_acoustic_exact_Pa`、`p_rms_acoustic_exact_Pa` | 精确声分支 `d2*C2*exp(-sigma2*x)` 及 Eq. (24)，保留热耗散吸收；热层内不能替代完整压力 |
| `p_hat_near_approximate_Pa`、`p_rms_near_approximate_Pa` | Eqs. (21)/(22) 平面波近似；需薄热层、热层外、`x <= R0` 且声吸收可忽略 |
| `p_rms_far_Pa` | Eq. (23)，近似近场 RMS 乘 `R0/x`，中心轴 `x > R0`；只给幅值，不提供远场复相位或吸收修正 |
| `rayleigh_distance_m`、`in_near_field` | `R0=s*f/c0` 与 `x <= R0` 的几何检查；不代表全部近似条件满足 |
| `delta_T_m`、`acoustic_decay_exponent` | 近似热层尺度 `sqrt(2*alpha/omega)` 与精确 `Re(sigma2)*x`，用于检查热层位置和声吸收 |

所有复幅采用 `exp(+iωt)`，RMS 为 `abs(hat)/sqrt(2)`；不额外作论文图中可能存在的 3.01 dB 调整。
近似还要求 `omega << P0/(rho0*alpha)`。近远场值不自动拼接；上例距离在 Rayleigh 距离之外，
应使用远场 RMS，近场值仅作为公式对照。频率或距离扫描可由调用方循环调用此入口。
默认保持 `TABLE1_AIR` 原常数；可显式传入 `gas=LimGas(...)`，与 LBM 对照时使用 `matched_gas(mapping)`。
本入口仅求交流响应，不由交流复幅推断 Eq. (15) 的直流温升，也不生成器件 `validation_passed`。
所有输出均为线性小扰动谐波响应；“精确”指上述线性方程的平面解。
精确分支沿用现有特征根算法，回归覆盖 10 Hz–1 MHz；微 Hz 量级存在已观测到的浮点消减误差。

需要单独计算时，可用 `near_field_pressure_hat_approximate` 或 `acoustic_pressure_hat_exact`。
这两个接口允许 `x_m=0`；已有 `near_field_pressure_rms_exact` 仍返回完整平面解的 RMS，
Eq. (24) 应使用声分支的模除以 `sqrt(2)`。

**两条独立实现之间的恒等关系**（`verification/test_lim2013.py` 守护）：
本仓链路 ÷ 论文 Eq. (22) = `γ·P0 / [(γ−1)·ρ0·cp·T0]`，与频率无关。用 Table 1 常数时该值
= 1.0130，即论文 Table 1 自身 `ρ0cpT0(γ−1)/γ = 100025 Pa` 与 `P0 = 101325 Pa` 的差；改用
本仓常数则为 1.000943，六位吻合。

论文 Table 1 另有一处 0.2% 差异：`alpha = 2.25e-5` vs `kappa/(rho0*cp) = 2.2451e-5`；
`with_alpha_from_kappa()` 可消掉。

未迁入：`continuum_1d_time.py`、`nonlinear_nsf_1d.py`（时域/非线性参照，不在当前链路内）。

2026-09-18 原始 PDF 核查：文章页码为 **5451–5461**，DOI `10.1016/j.jsv.2013.05.020`。
式 (19) 末项的印刷负号与精确出射分支、式 (21) 不一致；热流向外为正时使用
`p_hat ≈ (gamma−1) q_hat/c0`，位置 x 的传播相位另乘。
近场 LBM 对照的统一幅值定义、EOS 常数、网格结果与适用范围见
[专项报告](../docs/LIM_NEARFIELD_20260918.md)。
