# reference/

独立参照与常数。参照只用于对照，不构成 LBM 器件认证。

| 文件 | 用途 |
|---|---|
| [continuum_1d_freq.py](continuum_1d_freq.py) | 一维半解析热声参照（`solve_level_C_frequency`），组合热导纳与解析声压剖面；不是独立离散的完整 NSF 求解器 |
| [lim2013.py](lim2013.py) | Lim/Tong/Li JSV 2013 悬空 CNT 薄膜热声源的精确解（Eqs. 8-16）与近似解（Eqs. 17-23）+ Rayleigh 距离；近/远场对照的权威参照 |
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
