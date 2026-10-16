# CNT 薄膜热声 LBM

本项目研究电热驱动下 CNT 薄膜的热声响应：**功率输入 → 膜温变化 → 膜–气换热 → 声压响应**。
采用“公共计算库 + 独立算例”的组织方式：公共库实现物理模型、数值推进和分析，`Compute/<专题>/` 保存工况配置、必要的短脚本和结果。

当前气体计算只支持 **D2Q37 + 显式 `transport.mode: local`**，默认使用 Sutherland 输运。
已提供闭域膜气耦合、开放平面与有限条膜研究入口；开放平面限于当前小振幅正入射设置，有限条膜及远场交接需逐工况验证。
回归或单个算例的验证不代表完整 CNT 器件链路已验证。工作规则见 [AGENTS.md](AGENTS.md)。

## 目录介绍

| 目录 | 用途 |
|---|---|
| [core/](core/README.md) | D2Q37 格子、平衡态、碰撞、流步、宏观量、单位与物性映射，以及气体求解器 |
| [boundary/](boundary/README.md) | 热壁、开放出口、边界条件及实际交换审计 |
| [coupling/](coupling/README.md) | 面功率驱动、膜温 ODE、膜–气保守耦合与膜内导热 |
| [phase3_interfaces/](phase3_interfaces/README.md) | 热流投影、单位交接、壁面状态和复幅值约定；名称沿用历史阶段 |
| [simulation/](simulation/__init__.py) | 配置展开、采样、批量运行、四文件输出、版本记录及只读重分析 |
| [postproc/](postproc/README.md) | 谐波与漂移拟合、幅相比较、稳定性及守恒分析 |
| [reference/](reference/README.md) | 热导纳、膜 ODE、半解析热声场与 `solve_lim2013_frequency` 近/远场公式参照 |
| [farfield/](farfield/README.md) | 二维 Kirchhoff/Hankel 外推、解析紧致源及有限条膜交接 |
| [configs/](configs/README.md) | 公共气体基线与独立远场验证夹具 |
| [scripts/](scripts/README.md) | 公共命令行入口、专项验证及绘图编排 |
| [verification/](verification/README.md) | 公共回归测试与输运测量工具 |
| [Compute/](Compute/) | 专题配置、短脚本及按批次保存的计算结果 |
| [Paper/](Paper/) | 参考论文与提取资料 |
| [Time-domain/](Time-domain/) | 时域热声与尺度分离研究的文献综述、离散推导和计划材料 |
| [X/](X/) | E1 局部热声研究草稿 |
| [docs/](docs/README.md) | 专题说明与历史验证材料 |
| [results/](results/) | 历史结果，保留原位 |
| [k-wave-toolbox-version-1.4.1/](k-wave-toolbox-version-1.4.1/) | 本地 k-Wave 工具箱，供 MATLAB 传播交接使用 |

根目录 [config_io.py](config_io.py) 负责 YAML 与 `include` 加载；Python 版本和依赖见
[.python-version](.python-version)、[requirements-lock.txt](requirements-lock.txt)。

`Compute/` 的现有专题：

| 专题 | 内容 |
|---|---|
| [open_planar/](Compute/open_planar/study.yaml) | 开放平面给温、脉冲及 E1 功率冷启动算例 |
| [film_to_sound/](Compute/film_to_sound/study.yaml) | 功率–膜温–实际换热–直接声压的局部平面演示与有限膜研究 |
| [finite_strip/](Compute/finite_strip/study.yaml) | 有限条膜二维近场、PML 边界比较及 k-Wave 交接实验 |
| [lbm_acoustic_handoff/](Compute/lbm_acoustic_handoff/study.yaml) | LBM 近场时序收敛与声学交接研究 |
| [multi_grid/](Compute/multi_grid/study.yaml) | 均匀/两级网格对照与 [九矩状态表示](Compute/multi_grid/moment_study.yaml) 实验 |
| [lim2013/](Compute/lim2013/study.yaml) | Lim 2013 线性频域公式计算与 SPL 对照 |

## 常用工况参数

### 公共空气基线

来源：[gas_air_10k_d2q37_qmatched_local_transport.yaml](configs/gas_air_10k_d2q37_qmatched_local_transport.yaml)。
文件名中的 `10k` 不设置驱动频率；专题可以显式覆盖物性，实际输入以展开后的 `effective.yaml` 为准。

| 配置字段 | 基线值 | 单位 |
|---|---:|---|
| `physical.T0_K` | 300 | K |
| `physical.p0_Pa` | 101325 | Pa（名义参考值） |
| `physical.rho0_kg_m3` | 1.177 | kg/m³ |
| `physical.c0_m_s` | 347 | m/s |
| `physical.gamma` | 1.4 | — |
| `physical.cp_J_kgK` | 1003.408333 | J/(kg·K) |
| `physical.kg_W_mK` | 0.0263 | W/(m·K) |
| `physical.nu0_m2_s` | 1.57×10⁻⁵ | m²/s |
| `lattice.dx_m` | 2.61175×10⁻⁶ | m |
| `lattice.dt_s` | 7.44047619×10⁻⁹ | s |

`transport.model: sutherland` 随温度更新 μ、k；`model: constant` 仅固定 μ、k，局部 ν、α、τ 仍随状态计算。
SI/LU、松弛时间与热流系数统一使用 [core/unit_mapping.py](core/unit_mapping.py)。

### 常用算例设置

下表为已有配置的输入设置，未表示这些工况全部通过验证。完整参数与扫描点见对应配置。

| 配置 | 主要工况 |
|---|---|
| [open_planar/study.yaml](Compute/open_planar/study.yaml) | 给温基准：50 kHz、温幅 0.1 K、域长 1 mm、6 周期、渐启 2 周期；网格 `refine=1/2/4`，另有 75/100 kHz 点及脉冲组 |
| [open_planar/e1_study.yaml](Compute/open_planar/e1_study.yaml) | 功率/声频 20 kHz，平均 1.13 W，面积 9×10⁻⁴ m²；`C_A=0.0007 J/(m²·K)`（假设），双侧合计 `G=57.8 W/(m²·K)`；`refine=4`、域长 1 mm、最多 24 周期 |
| [film_to_sound/study.yaml](Compute/film_to_sound/study.yaml) | 同上功率、面积、面热容与热漏；`refine=2`、域长 1 mm、最多 24 周期，并设脉冲检查 |
| [finite_strip/study.yaml](Compute/finite_strip/study.yaml) | 膜宽 1 mm，上半域 2 mm×0.2 mm；20 kHz、平均面功率 1255.56 W/m²，`C_A=0.012888 J/(m²·K)`、`G=0`；功率网格 dx=1.305875 μm，PML 厚 0.08 mm；脉冲另用 dx=10 μm 和大域对照 |
| [multi_grid/study.yaml](Compute/multi_grid/study.yaml) | 20 kHz、`refine=4`、域长约 1 mm；两级空间/时间比 2；给温 0.1 K、6 周期，功率沿用 E1 设置、固定 20 周期；另有双向接口脉冲和热阶跃 |
| [lim2013/study.yaml](Compute/lim2013/study.yaml) | 线性频域参照：声频 20 kHz、1.13 W、面积 9×10⁻⁴ m²；`c_s=0.012888 J/(m²·K)`、单侧 `β₀=28.9 W/(m²·K)`；距离 6/8/15/40/45 mm，SPL 参考压力 20 μPa |

有限条膜和 Lim 配置的面热容由采用的 `ρ_f·cp_f·h_f` 得到，不是 E1 实测值；与开放平面的假设面热容应分开。
论文工况与缺失参数的核对见 [工况参考表.md](工况参考表.md)。

常调字段包括 `mode`、`frequency_Hz`、`refine`、`length_m`、`cycles`、`probes_m`；
给温使用 `amplitude_K`、`ramp_cycles`，功率模式使用 `power`、`film`，不混入给温参数。
平面入口的 `refine` 同比缩小基线 dx、dt，实际域长和探针位置可能因格点取整略有变化。

- `frequency_Hz` 是壁温或功率/声学频率；纯交流焦耳热驱动的电频为其一半。
- 面功率单位为 W/m²，面积是单侧投影面积；上半域热流以膜到气体为正。对称双侧膜气耦合的单侧热流因子为 2，`G` 已是双侧合计，与单侧 β₀ 比较时取 `G=2β₀`。
- 数组为 `f,g=(ny,nx,Q)`，向量分量为 `(x,y)`；复幅采用 `exp(+iωt)`。区间平均热流按中点拟合、按区间求和积分。
- 传导热流矩、实际壁面能量交换、热流换算声压与气体直接声压分别报告；保守耦合使用实际壁面交换。

## 常用运行入口与输出

从项目根目录使用 Python 3.14.2 与项目 `.venv`。重建环境时用 `py -3.14 -m venv .venv`，
再执行 `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`。
新计算使用不存在的批次/算例名，先检查展开配置、步数和输出路径：

```powershell
# 开放平面：检查整批配置，再选择一个脉冲运行
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/open_planar/study.yaml --batch new_batch --dry-run
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/open_planar/study.yaml --batch new_batch --case pulse_r1
# 有限条膜与 Lim 参照使用专题入口
.\.venv\Scripts\python.exe -m Compute.finite_strip.run --batch new_batch --dry-run
.\.venv\Scripts\python.exe -m Compute.lim2013.run --batch new_batch --dry-run
```

平面入口不传 `--case` 会执行全部算例；Python 公共入口为
`simulation.prepare_case`、`run_case`、`run_study`、`analyze_case`，最后一项只读保存的数据。
有限条膜功率计算先检查同批次边界证据，后续交接仍有独立门禁；调用流程见专题脚本。

新开放域算例位于 `Compute/<专题>/<批次>/raw/<case>/`，包含：

| 文件 | 内容 |
|---|---|
| `effective.yaml` | 展开的物性、网格、驱动、采样及分析参数 |
| `manifest.json` | 执行状态、Git 版本、未提交状态、公共源码哈希及运行环境 |
| `trajectory.npz` | 探针、剖面和边界收支时序 |
| `summary.json` | 分析指标与 `validation_passed` |

正式图表放同批次 `final/`；`raw/`、`final/` 结果本地保留，专题配置和脚本纳入 Git。
Lim 等独立参照入口采用各自输出格式。运行状态与验证通过分别判断：平面出口脉冲总判据当前仅为反射比，
热声总判据不含热流幅相误差门限；网格、域长、二维传播与器件适用性需另行核查。
