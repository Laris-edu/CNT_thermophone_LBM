# CNT 薄膜热声换能器 LBM（核心链路仓）

从旧仓 `E:\JFM2024` 迁出的最小可运行集：只保留**功率 → 膜温 → 热流 → 复声压**这条已通过回归的链路，以及守护它的验证。论文参考目录、历史阶段文档、已退役速度集/碰撞分支和一次性审计脚本留在旧仓。

## 链路与代码

| 环节 | 代码 | 说明 |
|---|---|---|
| 功率 `P_in(t)` | [coupling/drive.py](coupling/drive.py) | 正弦/阶跃/高斯驱动，SI 面功率，允许有符号功率 |
| 膜温 `T_s(t)` | [coupling/film_ode.py](coupling/film_ode.py)、[coupling/conservative.py](coupling/conservative.py) | `C_A·dT_s/dt = P_in − 2q″_g`；热壁重构与膜储能在同一步联立求解 |
| 热流 `q″_g` | [boundary/wall_thermal_mass_neutral.py](boundary/wall_thermal_mass_neutral.py)、[phase3_interfaces/heat_flux_extraction.py](phase3_interfaces/heat_flux_extraction.py) | 取实际壁区间能量交换；row1 传导矩只作诊断 |
| 复声压 `p̂(Ω)` | [postproc/multiharmonic_fit.py](postproc/multiharmonic_fit.py) | 探头总压的趋势+多谐波联合拟合，区分均值/漂移/目标谐波 |
| 气体核 | [core/](core/) | D2Q37 多速格子、四阶 Hermite 平衡态、f-g 双分布（D=2, S=3）、局部输运 |
| 独立参照 | [reference/continuum_1d_freq.py](reference/continuum_1d_freq.py) | 一维 NSF 频域解，用于对照而非认证 |

链路入口是 `coupling.conjugate.run_levelc_predictor_corrector`（默认 `scheme="conservative"`）；接口约定、探针与限制见 [docs/CHAIN.md](docs/CHAIN.md)。

## 环境与运行

本目录的 `.venv`（Python 3.14.2）已按 `requirements-lock.txt` 装齐依赖，直接可用。
如需重建：

```powershell
cd E:\CNT_thermophone_LBM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --only-binary=:all: -r requirements-lock.txt
```

```powershell
python -m scripts.run_gas_smoke                      # 周期气体小扰动启动检查
python -m pytest -q                                  # 全量回归
python -m scripts.verify_nearfilm_chain --mode C --output results\case_C   # 完整近膜链路（长）
```

`results/` 已被 `.gitignore` 排除；运行输出、配置副本与哈希放在那里。

## 验证状态（随代码迁移，未新增结论）

- **已通过**：10 kHz 封闭对称双侧小扰动，域长不变、dx/dt 同时加密一倍后的 A/B/C 检查；膜温幅值/相位误差在 5%/5° 内，四个压力探针通过，膜气能量积分与谐波残差另有更严阈值。
- **未通过**：原网格的膜温幅相精度；row1 传导矩近壁点值仍偏低，已从能量交换路径移出，不能当作交换量。
- **未建立**：开放边界/半空间声压、有限宽薄膜的边缘与衍射、三维辐射；本仓的 `farfield/` 只到二维 Kirchhoff 频域外推，且目前只有制造解级验证。
- **Lim 2013 对照**：悬空 CNT 薄膜热声源的精确解与近似解见 [reference/lim2013.py](reference/lim2013.py)，复现入口是 `python -m scripts.verify_lim2013`。10 kHz 下 LBM 的**壁面热流**与 Lim 精确解差 0.00%（相位 +0.22°），声压由 `p = (γ−1)q/c₀` 换算，**不读压力探针**（闭箱压缩在本工作点比辐射声压大 93 倍）。参数口径表见 [reference/README.md](reference/README.md)。
- 回归阈值（剪切波相对误差 <0.5%、热扩散 <2%、声速 <0.2% 等）只在测试条件内成立，不是器件误差界限。
- 完整数字、配置与日志在旧仓 `results/`；本仓的等价性证据见 [docs/PROVENANCE.md](docs/PROVENANCE.md)。

## 目录

| 位置 | 内容 |
|---|---|
| [core/](core/) | 气体核：格子、平衡态、碰撞、流步、宏观量、单位映射、求解器 |
| [boundary/](boundary/) | 热壁：对称 mass-neutral（链路主用）、grad 与 dirichlet（A/B 与初始化） |
| [coupling/](coupling/) | 驱动、膜 ODE、保守交换、共轭入口、能量审计 |
| [phase3_interfaces/](phase3_interfaces/) | 热流提取、壁温单位契约、复幅相工具 |
| [postproc/](postproc/) | 趋势+多谐波联合拟合 |
| [reference/](reference/) | 1D NSF 频域参照、解析模型、热导纳、常数 |
| [farfield/](farfield/) | 2D Kirchhoff 外推与紧致源（未认证） |
| [verification/](verification/) | 链路回归与四个动态测量器 |
| [configs/](configs/) | local 基线、frozen 解析应力对照、Kirchhoff 夹具 |
| [scripts/](scripts/) | 启动检查、近膜链路驱动、Kirchhoff 验证 |
| [docs/](docs/) | 链路接口说明与迁移溯源 |

本次迁移的来源、删除项与等价性证据见 [docs/PROVENANCE.md](docs/PROVENANCE.md)。
