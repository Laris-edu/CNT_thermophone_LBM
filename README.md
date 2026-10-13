# CNT 薄膜热声 LBM

本项目研究电热驱动下 CNT 薄膜的热声效应：

**电功率输入 → 薄膜温度响应 → 固–流界面热交换 → 近场热声耦合 → 远场声压。**

项目按“公共计算库 + 独立算例”使用。公共代码负责物理模型、数值方法和通用分析；
`Compute/<专题>/` 放算例参数、必要的短脚本与计算结果。日常新增频率、功率、网格等
算例时，优先改算例配置，不复制或另写求解器。Agent 工作规则见 [AGENTS.md](AGENTS.md)。

当前气体模型只支持 **D2Q37 + 局部输运**。D2Q21、`transport.mode: frozen`、手动标定的固定热流系数与 Galilean 热流修正已退役。

## 1. 现有代码覆盖到哪里

各个环节已有实现，但尚未组成完整的开放域电热声器件计算链路。

| 研究环节 | 当前实现 | 使用边界 |
|---|---|---|
| 电功率输入 | `coupling.drive` 提供面功率信号，单位 W/m² | 输入的是热功率密度；没有自动求解电流、电压、电阻或焦耳热分布 |
| 薄膜温度 | `coupling.film_ode` 的集中参数膜温 ODE | 单一膜温与单位面积热容；不描述膜内温度分布 |
| 膜–气换热 | `coupling.conservative` 为闭域；`coupling.planar_open.step_open_film` 为开放域实际交换耦合 | 开放耦合为对称双侧单温膜，只计算上半气体域 |
| 开放域近场 | `PlanarOpenSolver` 支持给温或实际交换回调，直接恢复气体压力 | `nx=1` 平面几何；下壁全容纳漫反射、上侧线性正入射特征出口；适用于所验证的小振幅条件 |
| 远场外推 | `farfield.kirchhoff_2d` 和 `compact_source` | 独立二维外推核与解析紧致源模型；尚未完成与当前近场计算的全链路验证 |

开放域支持“**给定膜温 → 气体热声响应 → 直接声压**”与功率驱动的局部膜气耦合，并有独立声脉冲反射测试。
闭域热流换算声压、开放域直接恢复声压、二维远场外推声压，必须分别标明来源。
不能用其中一项通过测试，推断有限尺寸 CNT 器件的全部响应已验证。

有限条膜的实验入口为 `core.finite_strip_solver.FiniteStripSolver` 与
`simulation.finite_strip`。它复用 D2Q37 局部输运，底边膜内逐段漫反射、膜外镜面对称，
以 `coupling.finite_strip.solve_strip_film_midpoint` 解沿膜导热与双侧实际交换。
数组仍为 `(ny,nx,Q)`；气体域在上半空间，第三方向无限长，总能量为 J/m。
顶部和左右边界可显式选用 `pml_width_m` 声学 CFS-PML：四个方向记忆变量
通过平衡态增量耦合到原生 D2Q37 流步，读取当前局部密度与声速，独立记录
`pml_mass_lu` 与 `pml_energy_J_m`。它是待逐工况验证的声学吸收层，尚不宣称
完整热黏性离散系统严格匹配。旧局部法向边界/松弛层保留用于历史复现；
PML 与旧松弛层不能同时启用，原平面入口与默认值不变。
数值完成、正性和能量闭合不能代替二维返回误差及 20 kHz 近场验证。

`Compute/finite_strip/study.yaml` 配置 1 mm 膜宽、2 mm × 0.2 mm 上半域候选及大域脉冲对照，
用 `python -m Compute.finite_strip.run --batch <新批次> --dry-run` 检查展开参数；
`--case <名称>` 单独执行，`--compare-pulses` 比较保存的候选/参考脉冲与侧边隔离脉冲。
该专题吸收带厚 0.08 mm，位于固定上半域内：名义顶部从 y=0.12 mm 开始，侧面从
|x|=0.92 mm 开始（实际位置随完整网格单元取整）。三条上半交接线的
`handoff_contours_m` 按 `[半宽, 高度]` 配置为 (0.85,0.09)、(0.90,0.10)、
(0.91,0.11) mm；主线为第二条，宽 1.8 mm、高 0.1 mm。
`validation_probes_m` 为三条线外的直接压力探针，当前在 (0,0.115)、
(-0.915,0.05)、(0.915,0.05) mm。展开配置即检查实际膜边缘、嵌套关系和整个
双线性插值模板均在非吸收区。公共默认旧几何保留，旧批次的 effective.yaml 不改。
缩域脉冲用 sigma=20 μm、探针半径 50 μm，比较整个保存时序以捕捉更早的顶部返回；
仍是 dx=10 μm 的宽带检查，不能代替 r2 热黏性近场或 20 kHz 验证。
`pml_target_return` 是连续层阻尼设计值，不是计算得到或保证达到的返回误差。
功率计算先受脉冲门禁约束，之后仍需 20 kHz、网格、域长和闭合线稳定性检查。
`farfield.strip_surface` 按压力/切向速度偶对称、法向速度奇对称补齐闭合线；
`Compute/finite_strip/kwave_handoff.m` 将其转换为 k-Wave 质量源与力源。
该适配器的独立声源测试不等于 LBM 近场或有限条膜链路验证。
边界比较通过后，`python -m Compute.finite_strip.chain --batch <同一新批次>`
连续执行功率近场、`analyze.py` 的全部膜段/交接线稳定性和直接压力对照、合格源导出、
MATLAB 调用以及 45 mm 传播核验；任一门禁失败即停止并保留诊断。
边界证据核对保存数据哈希、几何/PML 参数和公共源码身份。
`validation_passed` 只覆盖运行状态与全场周期稳定，`source_validation_passed` 另含交接条件，
`propagation_validation_passed` 仅在真实 k-Wave 结果通过独立核验后赋值。

### 有限条膜的 64 线程服务器入口

在 Ubuntu 26.04 x86_64 的同一 Git 提交下建立 Python 3.14 虚拟环境，并按
`requirements-lock.txt` 安装依赖；只部署代码与配置，历史结果、Windows `.venv`、
MATLAB 和 k-Wave 留在本机。服务器只运行 Python LBM，普通按量实例适合正式长算。
现有异常输出不是可恢复检查点；抢占回收可能丢失未完成时序，`tmux` 仅防止 SSH 断线。

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m Compute.finite_strip.cloud --batch cloud_001 --stage boundary --dry-run
.venv/bin/python -m Compute.finite_strip.cloud --batch cloud_001 --stage boundary
.venv/bin/python -m Compute.finite_strip.cloud --batch cloud_001 --stage power --dry-run
.venv/bin/python -m Compute.finite_strip.cloud --batch cloud_001 --stage power
```

每次使用不存在的新批次名。入口在新 Python 子进程导入 Numba 前设置
`NUMBA_NUM_THREADS=64`、`CNT_LBM_PARALLEL_MIN_CELLS=131072`，并限制 BLAS 线程。
公共并行门槛默认仍为 262144 格；本专题 r2 为 234396 格，因此云入口会选择并行。
`--threads 16` 可用于本机检查，正式运行拒绝请求超过进程可用 CPU 数的线程；
`--dry-run` 可以在本机展示 64 线程配置，但不会运行内核或宣称硬件可用。
manifest 的 `runtime_started` / `runtime` 记录实际线程数、线程池上限、线程层、
格点门槛、调度选择、CPU 可用范围及依赖版本。CPU 线程数不是性能保证，膜温求解、
部分 NumPy 操作和 I/O 仍有串行开销；首次 JIT 与稳定运行耗时需分开报告。
云入口检查保存的验证数据，失败返回非零退出码并保留诊断；子进程正常结束不等于通过。
功率阶段的状态/周期门禁与原始 `validation_passed`、后续源交接/传播门禁分开报告。

边界和功率计算必须在同一公共源码身份下运行；缩域不能复用旧域的通过证明。
将整个新批次的四组边界原始数据、功率四文件和 `final/boundary_validation.json`
按原始字节传回本机，再执行以下命令。此阶段读取保存的配置和云端源码身份，
不会重跑 LBM；合格后才在本机生成带本地工具箱路径的 MAT。
LBM 的直接探针仍使用保存的原坐标；k-Wave 的 50 μm 网格不能把这些近源点直接
吸附到声源线上。导出时把过近的传播核验点沿外法向方向移至距主矩形至少三格，
并检查吸附后仍有两格间距。当前三点为 (0,0.25)、(-1.05,0.05)、(1.05,0.05) mm，
仅用于独立 Kirchhoff 传播对照，不能称作这些位置的 LBM 直接验证。

```powershell
.\.venv\Scripts\python.exe -m Compute.finite_strip.chain --batch cloud_001 --stage propagate --matlab E:/Matlab/bin/matlab.exe --toolbox E:/CNT_thermophone_LBM/k-wave-toolbox-version-1.4.1
```

原始云端数据与本机分析的源码身份分别记录，避免 Windows/Linux 行尾差异被误当作
同一原始字节；不改写已保存的哈希或四文件。已有分析/传播文件仍拒绝覆盖。

用户明确要求先查看未验证传播时，可对已完成且周期稳定的保存源使用
`--stage propagate --diagnostic --matlab-threads 4`。该选项仅允许跳过
`kirchhoff_vs_direct` 的源比较失败，其余状态、几何、周期稳定和边界证据检查仍执行；
MAT 输入及输出保留 `source_validation_passed=0` 与显式诊断标记。
结果另存为 `power_r2_kwave_diagnostic_input.mat`、
`power_r2_kwave_diagnostic_output.mat` 和 `propagation_diagnostic.json`，
原近场诊断不覆盖。诊断运行退出码 0 仅表示传播执行完成；各项误差和
`propagation_validation_passed` 仍按原门限如实记录，不代表器件或交接验证通过。

## 2. 公共代码导航

| 目录 / 文件 | 封装的职责 | 通常何时使用 |
|---|---|---|
| [core/](core/README.md) | 格子、平衡态、碰撞、流步、宏观量、单位与物性映射、气体求解器 | 推进气体状态；普通算例调用求解器，不直接改碰撞核 |
| [boundary/](boundary/README.md) | 周期域热壁、质量中性热壁、开放域热壁与出口、边界交换审计 | 改变边界类型或研究界面传热 |
| [coupling/](coupling/README.md) | 功率信号、膜 ODE、膜–气联立推进、能量审计 | 从功率求膜温和单侧热流 |
| [phase3_interfaces/](phase3_interfaces/README.md) | 热流法向投影、SI/LU 转换、壁面状态与复幅值约定 | 在求解器、耦合与分析间交接数据；目录名沿用迁移时命名 |
| [simulation/](simulation/__init__.py) | 配置展开、时间采样、批量运行、四文件输出、源码溯源、只读重分析 | 新的开放平面算例优先使用这一层 |
| [postproc/](postproc/README.md) | 多谐波与漂移联合拟合、幅相比较、开放域分析 | 从已保存时序提取结果，无需重算 LBM |
| [reference/](reference/README.md) | 热导纳、膜 ODE 闭式解、半解析热声场、Lim 2013 参照 | 独立对照、量纲和极限检查；不反向填充 LBM 的源项或边界 |
| [farfield/](farfield/README.md) | 二维 Kirchhoff/Hankel 积分与解析热泵送源强 | 单独研究外推核及其适用条件 |
| [configs/](configs/README.md) | 公共物性/格子基线，以及独立远场夹具配置 | 作为算例基础；不堆放每次扫描的配置副本 |
| [scripts/](scripts/README.md) | 公共命令行入口和专项验证、绘图编排 | 从终端运行；专题专用小脚本放对应算例目录 |
| [verification/](verification/README.md) | 自动回归与动态输运测量器 | 修改公共算法后检查已有行为 |
| [config_io.py](config_io.py) | YAML 加载与递归 `include` | 底层配置读取；合并是顶层覆盖，不是字段级递归合并 |

`docs/`、`Lim/`、`results/` 中的历史材料与结果保留原位，不是新算例的默认代码或输出位置。

### 气体内核如何衔接

1. [unit_mapping.py](core/unit_mapping.py) 建立 SI/LU 映射和参考态，每步由
   `local_transport_coefficients(mapping, rho, theta)` 生成逐格点松弛时间和热流系数。
2. [equilibrium.py](core/equilibrium.py) 构造四阶 Hermite `f` 平衡态和二阶 `g` 平衡态。
   `f` 承载质量、动量和平动能，`g` 承载内部能；当前为 `D=2, S=3, γ=1.4`。
3. [collision_smrt.py](core/collision_smrt.py) 使用二阶中心矩闭合，按实测非平衡偏应力松弛；
   热流做最小范数投影，并逐格修正质量、动量和总能量。没有正性裁剪。
4. [streaming.py](core/streaming.py) 负责传播。周期与法向开放流步是不同入口；
   开放边界越界链接必须由边界条件填入，不能周期回绕。
5. [macroscopic.py](core/macroscopic.py) 恢复密度、速度、温度、压力与热流。
   [GasSolver2D](core/solver.py) 编排周期碰撞–流步–边界回调；
   [PlanarOpenSolver](core/planar_open_solver.py) 通过 `step_temperature(T_midpoint_K)` 推进一步开放域。

公共代码作为仓库内模块导入，目前没有独立发布的 Python 安装包。命令统一从项目根目录执行，
不需要把算法复制到算例目录，也不需要在每个脚本里修改 `sys.path`。

### 关键参数与接口约定

- **输运路线：**必须显式写 `transport.mode: local`。主用 `model: sutherland`，
  按温度更新 μ、k，再由局部密度得到 ν、α。`model: constant` 仅固定 μ、k，
  仍按局部密度、温度更新松弛时间，供物性对照使用，不是已退役的 frozen 路线。
- **单位映射：**`core.unit_mapping` 是换算与输运参数的唯一来源。dx、dt、单位比例和求积温度
  在一次运行内固定；局部物性变化不意味着重设单位。cp 必须与 D/S 和温度单位一致。
- **热流系数：**使用局部 `h = 1 − 1/τ32` 和碰撞前读出系数 `1 − 1/(2τ32)`。
  元数据中的 τ 和热流系数只是参考态记录，不是独立可调的固定标定系数。
- **数组与方向：**`f,g = (ny,nx,Q)`，`c = (Q,2)`，速度与热流分量为 `(x,y)`。
  上半气体域的单侧热流以从膜流入气体为正。
- **传导矩与交换量：**`extract_wall_heat_flux(..., config=config)` 返回局部修正后的法向传导矩。
  不传配置时默认 D2Q37，但只返回原始中心能流矩，不能当作物理传导热流。
  保守膜气耦合使用实际热壁能量交换，不能用近壁某一行的传导矩替代。
- **双侧系数：**膜方程为 `C_A·dT_s/dt = P_in − gas_flux_factor·q_g − G·(T_s−T_ref)`。
  `q_g` 是单侧量，悬空对称双侧默认因子为 2；`G` 已是合计热漏。
  与 Lim 论文的单侧 β₀ 比较时取 `G=2β₀`。
- **时间与相位：**统一 `x(t)=Re[x̂ exp(+iωt)]`。拟合 `A cos(ωt)+B sin(ωt)` 时，`x̂=A−iB`。
  保守耦合 `q[1:]` 是前一区间平均热流，在中点拟合，累计能量用 `Σ q[1:]Δt`。
  开放域壁温同样在中点施加，气体压力在步末读取。
- **声压：**原始压力含均值和热漂移，分离后再提取谐波。Lim 对照用 `matched_gas(mapping)`
  匹配实际 LBM 状态方程，不直接用名义 `p0_Pa` 替代。
  频率参数是施加功率或壁温信号的频率，不自动等于电流频率。

旧 frozen 配置、缺少 `transport.mode` 的配置、手动热流系数和显式 D2Q21 都会报错。
原 `gas_air_10k_d2q37_qmatched_analytic_stress.yaml` 已移除；历史结果使用其记录的源码版本复现，
不能删掉报错字段后就视为同一物理模型。

## 3. 算例与结果放在哪里

```text
项目根目录/
  core/ boundary/ coupling/ ...       # 公共计算库
  configs/                           # 公共基线
  verification/                      # 公共回归
  Compute/
    my_case/                         # 一个专题，名称自行确定
      study.yaml                     # 物理参数、网格、扫描方案
      run.py                         # 可选：特殊驱动或编排，调用公共库
      plot.py                        # 可选：本专题绘图
      batch_001/
        raw/<case_name>/            # 可追溯原始数据
        final/                     # 按需生成的正式图表和汇总表
        _work/                     # 仅确有中间调试文件时创建
```

只改频率、温度幅值、域长、网格或探针时，一份 `study.yaml` 即可。
只有公共入口不能表达的算例逻辑才需要 `run.py`；它应编排公共函数，不复制碰撞、边界或单位换算。
功能确实被多个算例使用时再提取到公共库，避免每个算例增加一套通用代码。

`simulation.run_case` 的新开放域工作流固定输出四个文件：

| 文件 | 内容 |
|---|---|
| `effective.yaml` | 展开后的物性、网格、驱动、采样和分析参数 |
| `manifest.json` | Git 提交号、未提交状态、公共源码哈希、环境、执行状态、起止时间 |
| `trajectory.npz` | 探针时序、边界收支及按配置采样的剖面 |
| `summary.json` | 幅相、对照误差、守恒与稳定性指标、判据结果 |

代码统一由 Git 管理，正式计算前应提交公共代码和算例脚本；各算例单独保存实际参数。
运行程序只读版本信息，不自动提交、复制源码或生成 ZIP，也不创建 `Compute/_sources/`。

`manifest.json` 中，`git_commit` 记录提交号，`git_dirty` 表示整个仓库是否有未提交的变化
（含 Git 可见的算例脚本与其他文件）；无法取得状态时为 `null`，不冒充干净状态。
`source_clean` 只表示公共运行源码是否与该提交一致，`source_sha256` 记录这些文件的哈希。
算例脚本不在公共源码哈希范围内，应纳入同一 Git 提交，不能仅凭 `source_clean=true` 判断完整复现条件。
试算允许未提交改动，但哈希不能恢复改动内容；正式结果应对应明确保存的代码版本。

已有 `_sources/` 和历史 `source_snapshot.zip` 保留原位；重分析已有时序不读取或重建源码归档。
原 Python API 的 `source_cache` 参数仍兼容接收，但不再产生任何缓存或快照。
新运行拒绝覆盖已有算例目录。一般用新批次名；同批次可追加尚不存在的算例，但要清楚记录版本。
正常计算不自动生成额外 README、报告、截图或多份源码副本。
旧闭域与专项 `verify_*` 入口保留各自格式，四文件约定不自动适用于所有历史脚本。

## 4. 从配置运行算例

使用项目 `.venv`，版本与依赖记录在 [.python-version](.python-version)、
[requirements-lock.txt](requirements-lock.txt)。不依赖激活状态的调用方式是：

```powershell
cd E:\CNT_thermophone_LBM
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m scripts.run_compute --help
```

若换电脑后确需重建环境，先安装所记录的 Python 版本，再执行：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

例如将下面内容保存为 `Compute/my_case/study.yaml`。示例目录和批次仅作说明，不会自动创建：

```yaml
base_config: ../../configs/gas_air_10k_d2q37_qmatched_local_transport.yaml
defaults:
  mode: thermal
  frequency_Hz: 50000
  amplitude_K: 0.1
  refine: 4
  length_m: 0.001
  cycles: 6
  ramp_cycles: 2
  probes_m: [0.00012, 0.00030, 0.00060]
cases:
  - name: pulse_r1
    mode: pulse
    refine: 1
sweeps:
  - name: frequency
    parameters:
      frequency_Hz: [50000, 75000, 100000]
```

`base_config` 相对于 `study.yaml` 定位；`defaults` 与单算例参数递归合并。
`sweeps` 取列表的笛卡尔积，生成如 `frequency__frequency_Hz_50000` 的名称。
`refine` 同比缩小 dx、dt；实际域长和探针位置会因格点取整略有偏差，分析用记录的实际位置。

```powershell
# 先检查参数、步数、名称与目标路径，不推进计算、不写输出
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/my_case/study.yaml --batch batch_001 --dry-run
# 选择一个脉冲算例
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/my_case/study.yaml --batch batch_001 --case pulse_r1
# 单独运行一个热声扫描点
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/my_case/study.yaml --batch batch_001 --case frequency__frequency_Hz_50000
```

不传 `--case` 会顺序执行全部算例，计算异常会停止后续运行。完整热声扫描可能耗时较长。
采样与分析默认值由 [simulation/study.py](simulation/study.py) 定义，并完整写入 `effective.yaml`；
默认使用末两个周期窗口检查稳定性，保留末两周期剖面与逐步探针、收支数据。
给温算例还保存步末探针法向速度 `velocity_m_s`（正向为膜到出口）与同一剖面时刻的
`profile_velocity_m_s`；分别与 `pressure_Pa`、`profile_pressure_Pa` 同形同时间。

## 5. 在自己的 Python 脚本里调用

```python
from pathlib import Path
from simulation import prepare_case, run_case, analyze_case

config = prepare_case({"name": "pulse_r1", "mode": "pulse", "refine": 1})
case_dir = Path("Compute/my_case/python_example/raw/pulse_r1")
result = run_case(config, case_dir)  # 目标目录必须不存在
print(result["validation_passed"], result["reflection_ratio"])

# 从保存的数据重新分析：不推进 LBM，也不覆盖文件
rechecked = analyze_case(case_dir)
```

若放在 `Compute/my_case/run.py`，从根目录运行
`.\.venv\Scripts\python.exe -m Compute.my_case.run`；专题目录名应为有效 Python 模块名。
整批计算也可用 `simulation.run_study(study_path, batch_name, selected=[...])`。

| 进一步定制 | 公共入口 | 必须留意 |
|---|---|---|
| 自定义给温信号 | `PlanarOpenSolver.step_temperature(T_midpoint_K)` | 温度为 K，当前仅开放平面；不能调用其继承的普通 `step()` |
| 周期气体与体内输运 | `GasSolver2D.initialize_from_macro()`、`step()`、`get_macro()` | 宏观初始化用 LU；物理量通过 `solver.mapping` 换算 |
| 功率驱动膜气耦合 | `run_levelc_predictor_corrector(solver=..., params=FilmOdeParams(...), drive=..., n_steps=...)` | 默认 `scheme="conservative"`；当前只支持对称双侧闭域，不能直接接开放域求解器 |
| 幅相与谐波 | `postproc.multiharmonic_fit.fit_multiharmonic`、`postproc.diagnostics.fit_window` | 明确拟合窗口、漂移项、相位零点和采样时间位置 |
| 开放域时序 | `postproc.planar_open.analyze_thermal` / `analyze_pulse` | 参数口径应与有效配置一致 |
| 解析对照 | `reference.lim2013`、`thermal_admittance`、`film_ode` | 参照假设和物性需与比较对象匹配 |

`scripts.plot_planar_open` 是专项绘图器，需要 50 kHz、约 1 mm 域长、r=1/2/4 热声组和
r=1/4 脉冲组；不是任意单算例绘图入口。其他专题绘图放自己的 `plot.py`，
复用公共分析函数，将正式图表写入该批次 `final/`。

## 6. 怎样判断结果可信

**运行完成与数值判据通过是两件事。**`manifest.status` 记录
`running/completed/failed/interrupted`；`summary.validation_passed` 单独记录判据结果。
CLI 正常退出也不等于所有物理判据通过。

- 热声算例检查压力幅相、跨周期稳定性、热壁质量交换、逐步与累计质量/能量收支、布居正性。
  热流与参照的误差会报告，但目前未纳入热流幅相误差门限，不能声称它也因总判据通过而达标。
- 脉冲的 `validation_passed` 当前只代表反射比通过，守恒残差另行报告。
- 可捕获的异常/中断会保存状态和已采样数据；这些不是完整稳态结果，也不是可续算的 `f/g` 检查点。
  强制结束进程或断电可能留下 `running`，需核实是否真的完成。
- 膜 ODE 的右端代回残差只能检查自洽；独立能量核查需比较实际储能、输入功与边界交换。
- 解析源强或制造解只验证相应工具。`continuum_1d_freq` 是半解析参照，
  不是独立离散的完整 NSF 求解器；二维外推核通过制造解也不等于完整器件验证。

公共代码修改后运行对应回归；碰撞、单位映射、边界和耦合等跨模块变化运行全套：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

普通参数扫描运行算例自身检查即可，不为每个参数新建测试文件。
支持范围以实际实现和可复现测试为依据，历史阶段名或“certified”字样不能替代当前验证。

## 7. 开放域功率驱动与冷启动

`coupling.planar_open.step_open_film(solver=..., params=FilmOdeParams(...), T_s_K=..., drive=...)`
联立中点膜温与实际单侧壁面交换，返回步末膜温、中点壁温、输入/热漏区间能量及气体审计。
输入为 W/m²，当前要求 `gas_flux_factor=2`；`G` 是双侧合计附加热漏。膜状态由调用方在成功返回后更新。
`PlanarOpenSolver.step_wall_energy(solve_temperature)` 将试算温度到实际壁面区间能量的函数传给回调，
仅在回调及边界检查成功后提交一次气体状态。原 `step_temperature` 入口保持兼容。

功率算例使用独立的 `mode: power` 参数结构，不能混入 `amplitude_K` 或 `ramp_cycles`：
`power.mean_W`、`power.area_m2` 指定总平均功率和投影面积；`film` 指定面热容、合计热漏、参考及初始膜温。
`frequency_Hz` 是功率/声学频率；此模式采用纯交流焦耳热波形，电频为其一半：

$$
P(t)=\overline P[1-\cos(2\pi f_a t)],\qquad f_e=f_a/2.
$$

$$
C_A(T_s^{n+1}-T_s^n)=p_{\mathrm{in}}''(t_{n+1/2})\Delta t
-2\Delta E_{\mathrm{wall}}^n-G(T_s^{n+1/2}-T_0)\Delta t.
$$

`cycles` 是上限；默认第 6 周期开始诊断、第 8 周期起允许提前结束。四个两周期窗口的基频幅值极差
小于末窗口幅值的 1%，相位跨度小于 1°，同时通过两/四周期及线性/二次漂移拟合的幅相敏感性检查。
`stopping.stop_when_stable: false` 可用于固定时长的短回归。温度、压力、速度在步末采样，实际热流在区间中点拟合。
从冷启动起保存每周期 32 组气体剖面；完整原始时序支持只读重分析。

E1 配置入口为 `Compute/open_planar/e1_study.yaml`：平均 1.13 W、面积 9 cm²、电频 10 kHz、声频 20 kHz，
采用假设面热容 $7\times10^{-4}\,\mathrm{J/(m^2 K)}$、合计附加热漏 $57.8\,\mathrm{W/(m^2 K)}$。
气体配置记录论文表值与自洽物性的差异，缺失的面热容及黏性参数来源写入有效配置和结果。

```powershell
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/open_planar/e1_study.yaml --batch 20260921_e1 --dry-run
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/open_planar/e1_study.yaml --batch 20260921_e1
.\.venv\Scripts\python.exe -m Compute.open_planar.plot_e1 --case-dir Compute/open_planar/20260921_e1/raw/e1_power_f20000_r4
```

功率算例的 `validation_passed` 仅汇总当前网格和域长下的守恒、有限性、正性与有限窗口周期稳定性。
平均温升/漂移另报；300 K 线性参照仅作独立诊断，不作为完整冷启动的精确参照门限。
到时长上限未稳定时，`status=completed`、`validation_passed=false`；异常则保留部分数据及失败状态。
输出尚未验证有限膜辐射、网格/域长独立性、长期热稳态或暖背景下出口适用性。

## 8. 守恒两级开放平面基线

`core.two_level_planar_solver.TwoLevelPlanarOpenSolver` 实现固定 `nx=1`、下细上粗、
空间与时间比均为 2 的 cell-centered merge/explode 基线（算法来源：
[Coreixas–Latt §3.2](https://arxiv.org/html/2504.04465v3)）。沿用 D2Q37、局部输运、
实测偏应力和 τ22 迹通道；本入口仅接受 `diagnostic_zero` 体黏度。
两层各自调用公共单位映射与输运本构，不叠加非平衡缩放或滤波。

在 study 算例参数中加入：

```yaml
grid: {type: two_level, ratio: 2, interface_m: 0.000125364}
sampling: {probe_interpolation: linear, sync_fine_steps: 2}
```

`refine` 仍定义最细网格；`numerics.ny` 在有效配置中为全域等效细格数。
`grid_layout` 记录物理细/粗格、六行影子格、实际域长、同步步数及格点更新量，
`level_mappings` 保存两层 SI/LU 映射。接口必须位于公共粗细单元面，粗层也必须恰好容纳整数格。
未指定 `grid` 的算例保持均匀网格，旧入口默认最近格点采样。
`probe_interpolation: linear` 在共同物理坐标对质量、动量和总能量插值后恢复观测量；不改求解状态。
均匀对照指定 `sync_fine_steps: 2`，与两级同步气体观测频率一致。

每个宏步先碰撞/迁移粗格、处理顶部边界，将向下布居展开到细影子格；随后两次仅碰撞物理细格，
影子只迁移，每次处理底壁；最后成对平均向上影子布居，只填粗迁移缺失链接。
`step_temperature(T)` 接受中点常量或时间函数；`step_wall_energy(callback)` 的回调为
`callback(exchange_J_m2, substep_index, midpoint_time_s)`，每宏步调用两次。
影子不进入储能、探针、物理剖面或格点碰撞计数。宏步全部检查成功才提交物理状态、时钟和审计。
`coupling.planar_open.step_two_level_open_film` 返回两条细步膜记录及一条同步气体审计；
与均匀入口共用 `solve_film_midpoint`，使用实际漫反射壁交换。

四文件契约不变，新增轨迹口径为：

- `t_s`：气体同步端点；`sync_fine_step`：对应细步累计数。
- `film_t_s`：细步膜温端点；`heat_time_s`：细步热流中点。
- `wall_energy_J_m2`、输入功和热漏：已积分的细区间能量；按区间求和，不再乘 dt 或梯形积分。
- `z_m`、`cell_width_m`、`level_index`：仅物理单元；气体储能与 SI 面质量 `mass_kg_m2` 均按宽度加权。
  兼容 `mass_lu` 统一为面质量除以 `rho_scale * finest_dx`。
- `interface_exchange[sync, side, moment]`：上下两侧独立实际链接交换及其和；四矩为质量、x/y 动量、总能量，
  SI 单位依次 kg/m²、kg/(m·s)、kg/(m·s)、J/m²，正号为进入该块。
  `block_conserved` 独立保存两块储量。均匀脉冲/阶跃记录同位置内部面的交换。
- `level_steps=[fine,coarse]`、`cell_updates`：实际成功推进工作量，分别记录预热、推进及总耗时。
  失败轨迹截止最后完整同步点。

`mode: interface_pulse` 支持独立双向接口实验；它的反射窗口由源、接口与探针决定，
不改变旧 `mode: pulse` 的出口反射含义。`mode: thermal_step` 检查热扰动跨界面的实际交换和传导矩。
给温/功率仍复用现有分析入口；功率独立核查细步膜储能、同步气体储能及膜气总储能。
`validation_passed` 含义保持原约定，新增均匀对照误差另写 `final/comparison.json` 中的 `comparison_passed`。
运行完成、旧判据通过、新比较目标通过三者分别报告；固定 20 周期冷启动不自动等于周期稳定。

```powershell
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/multi_grid/study.yaml --batch new_batch --dry-run
# 先选择 pulse_up/down 与 step 的六个算例，检查守恒、有限性、正性后，再运行 thermal 和 power 对照。
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/multi_grid/study.yaml --batch new_batch --case pulse_up_two_level pulse_up_uniform pulse_down_two_level pulse_down_uniform step_two_level step_uniform
.\.venv\Scripts\python.exe -m Compute.multi_grid.compare --batch Compute/multi_grid/new_batch
.\.venv\Scripts\python.exe -m Compute.multi_grid.plot --batch Compute/multi_grid/new_batch
```

该实现是静态一维保守接口的研究基线；方法创新性需以相对此基线的实质改进和误差证据论证。
它尚不覆盖动态自适应、多级网格、多维膜器件、长期热稳态或一般暖背景出口。

## 9. 热—声分场研究：同网格九矩入口

`core.moment_planar_solver.MomentPlanarOpenSolver` 是现有正则化 D2Q37 算法的九矩状态表示。
它保留局部输运、实测应力及其迹、真实热壁交换和原膜中点耦合。目前只接受均匀网格、`nx=1`；
**不是已经完成的热细声粗求解器，也不代表研究对照 C/D。**
算例参数增加 `state_representation: moments9`，并指定 `sampling.sync_fine_steps: 2`；
缺省 `populations` 保持原入口。有效配置选择项在 `numerics.state_representation`。
`moments9` 与 `grid` 同时出现会报错，不会静默退回均匀网格。

令 $F=(f,g)$，固定线性提取算子 $B$ 定义

$$
z=BF=(\rho,j_x,j_y,E,M_{xx},M_{xy},M_{yy},J_{E,x},J_{E,y}),\quad
M_{ab}=\sum_i c_{ia}c_{ib}f_i,
$$
$$
E=\sum_i(\tfrac12|c_i|^2f_i+g_i),\qquad
\boldsymbol J_E=\sum_i\boldsymbol c_i(\tfrac12|c_i|^2f_i+g_i).
$$

从这九个量可以恢复原碰撞使用的全部量：

$$
u=j/\rho,\quad p=(\gamma-1)(E-\tfrac12\rho|u|^2),\quad \theta=p/\rho,
\quad P=M-\rho uu,\quad\Pi=P-pI,\quad q_{raw}=J_E-Eu-Pu.
$$

原碰撞的局部输运只依赖 $\rho,\theta$，非平衡重构只依赖 $\Pi,q_{raw}$。
故存在同一重构算子 $\mathcal C$，使 `collide(F)` 等价于 $\mathcal C(BF)$；
九矩推进为 $z^{n+1}=B\,\mathcal W\,S\,\mathcal C(z^n)$，$S$ 是原流步、
$\mathcal W$ 是原壁面/出口填充。这里是精确算术下的离散因式分解，实际比较允许浮点累积差，
不是把连续 NSF 方程的截断误差当作离散等价证据。65 维 $\ker B$ 在现有碰撞中被消去，
流步可以重新产生它，但下一次碰撞仍只需九矩。

公共接口位于 `core.kinetic_moments`：

| 接口 | 数据含义 |
|---|---|
| `extract_moments(f,g,lattice=None)` | 从真实布居提取末轴为 9 的原始矩 |
| `recover_moment_macro(z,mapping)` | 恢复现有 `MacroState`，含平动、内部及总能量 |
| `collide_moments(z,mapping,lattice=None,backend='numpy')` | 共用原碰撞重构，返回真实碰撞后布居 |
| `heat_flux_moments(z,mapping)` | 乘公共局部半步读出因子的传导矩；不是壁面交换 |
| `pressure_entropy_coordinates(z,mapping)` / `moments_from_pressure_entropy(coords,mapping)` | 可逆压力—速度—熵—应力—原始热流坐标 |

后一坐标顺序为 $(p,u_x,u_y,\sigma,\Pi_{xx},\Pi_{xy},\Pi_{yy},q_{raw,x},q_{raw,y})$，其中
$\sigma=\ln(p/p_0)-\gamma\ln(\rho/\rho_0)$，参考态来自实际 mapping。
它是热力学坐标，不是非平衡布居的离散 H 熵。正反变换拒绝非法密度/压力，不做裁剪。
`MomentPlanarOpenSolver.moments` 持久保存 `(ny,1,9)`，`f/g` 保持 `None`；每步仍暂存 37 方向的
碰撞后及流步布居并检查真实正性。因而 9/74 只是持久气体状态的大小比，不能代替峰值内存或实测加速。

四文件生命周期保持不变；新增 `state_storage_bytes`、`representation_equivalence_only`，
九矩轨迹另存 `moment_state_final`。它是最终原始矩，不能当作已实现续算协议。
同步入口的失败恢复涵盖气体、时钟、审计和未提交观测；膜温只在成功同步后写入历史。
`PlanarOpenSolver.checkpoint()/restore()` 与九矩/两级同名方法用于这一事务，副本归调用方所有。

复现入口为 `Compute/multi_grid/moment_study.yaml`，保持原脉冲、阶跃、给温和功率的物理参数。
先执行双向脉冲和阶跃，再根据证据决定是否开展周期长算例：

```powershell
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/multi_grid/moment_study.yaml --batch new_moment_batch --dry-run
.\.venv\Scripts\python.exe -m scripts.run_compute --config Compute/multi_grid/moment_study.yaml --batch new_moment_batch --case pulse_up_populations pulse_up_moments9 pulse_down_populations pulse_down_moments9 step_populations step_moments9
.\.venv\Scripts\python.exe -m Compute.multi_grid.compare_moments --batch Compute/multi_grid/new_moment_batch --pair pulse_up pulse_down step
```

专题比较只读四文件，检查相同源码、物理配置与时间坐标；不平移波形、不补偿幅相。
逐元素数值等价目标单独记录为 `rtol=1e-10, atol=1e-12`（各数组自身单位）；
它与原物理 `validation_passed`、1%/1° 波形目标分开。
近零热流可能由大能流项相消产生，必须同时查看绝对误差和信号归一化误差，不能只看一个布尔值。
阶段性批次可用 `--pair` 选择已完成的对子，未运行组不计入通过。
历史基线的只读定位与可选移动接口短实验由 `Compute.multi_grid.audit_interface` 执行，
`--output-batch` 必须为新目录；`--run-moved-interface --interface-m ...` 才会启动计算。

**独立网格的方程关仍未通过。**同一 $\rho,j,E$ 可有不同应力或热流，并在下一步产生不同宏观量，
这一反例已纳入回归；不能只保留四个宏观量，也不能把 `f/g` 分别命名为声/热。
线性参考态附近有

$$
\frac{\rho'}{\rho_0}=\frac{p'}{\gamma p_0}-\frac{\sigma'}\gamma,\qquad
\frac{T'}{T_0}=\frac{\gamma-1}{\gamma}\frac{p'}{p_0}+\frac{\sigma'}\gamma.
$$

连续能量余额给出 $D_t p+\gamma p\nabla\cdot u=-(\gamma-1)(\Pi:\nabla u+\nabla\cdot q)$，
$D_t\sigma=-(\gamma-1)(\Pi:\nabla u+\nabla\cdot q)/p$。
热流同时驱动压力和熵；近等压热层仍需热膨胀速度和局部输运反馈。
把 $p,u,\Pi$ 直接粗化而只细化 $\sigma,q$，尚缺细尺度密度/速度/应力修正及守恒通量闭合。
这些连续式解释耦合，不能用它们的残差替代离散独立储能审计。
连续式中的 $\Pi,q$ 是物理应力通量和传导热流，不能直接用碰撞前原始非平衡矩代替。
因此当前不进入独立分场网格、组合加密或二维性能宣称；九矩入口是用于推导与误差隔离的工具。

以下文献约束研究定位，而非本实现的新颖性证明（检索核对日期 2026-09-29）：

| 已有路线 | 本研究需要另外解决的问题 |
|---|---|
| [Dorari 等，2018，分网格/分时间步耦合输运](https://doi.org/10.1016/j.apm.2018.06.023)，用于枝晶的动量、溶质和热输运 | 已有“多物理不同尺度使用不同网格/步长”的先例；不能将一般构想视为首次提出 |
| [Nabavizadeh 等，2021，自然对流](https://www.mdpi.com/2311-5521/6/4/148)，热与流使用不同网格/步长 | 不可压 Boussinesq 假设不能直接提供可压热致声源与压力功闭合 |
| [Wissocq 等，2022，守恒分离模型](https://doi.org/10.1063/5.0083377)，利用 LBM 通量构造守恒总能量离散 | 熵特征与能量守恒必须一起处理；本项目需进一步处理独立网格和真实膜气交换 |
| [Zhao 等，2023，PRECCINSTA 热声燃烧](https://doi.org/10.1016/j.compfluid.2023.105898) | 已有 LBM 热声与非均匀网格案例，不能声称首次将二者结合 |
| [矩空间信息传递的多尺度框架，2026](https://doi.org/10.1016/j.cpc.2026.110025)，用于激光诱导合成 | 已有多物理尺度分离与矩投影先例；本文研究目标需以微弱声压、能量交换和额外成本收益区分 |

上述文献并不能证明本项目独立网格方案已经闭合。数值结论以批次 `final/` 数据为准，
不将九矩状态压缩、可逆坐标或局部恒等式作为完整方法创新或器件验证。
