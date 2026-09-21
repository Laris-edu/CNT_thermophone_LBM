# CNT 薄膜热声 LBM

本项目研究电热驱动下 CNT 薄膜的热声效应：

**电功率输入 → 薄膜温度响应 → 固–流界面热交换 → 近场热声耦合 → 远场声压。**

项目按“公共计算库 + 独立算例”使用。公共代码负责物理模型、数值方法和通用分析；
`Compute/<专题>/` 放算例参数、必要的短脚本与计算结果。日常新增频率、功率、网格等
算例时，优先改算例配置，不复制或另写求解器。Agent 工作规则见 [AGENTS.md](AGENTS.md)。

当前气体模型只支持 **D2Q37 + 局部输运**。D2Q21、`transport.mode: frozen`、
手动标定的固定热流系数与 Galilean 热流修正已退役。

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
