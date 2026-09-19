# CNT 薄膜热声 LBM

共用物理代码，通过参数配置建立算例；正式计算放在 `Compute/<专题>/<批次>/`。
当前优先支持 **给定膜温 → 单侧开放空气 → 直接声压**。原有闭域功率耦合和 Lim 热流换算入口保留。

## 能计算什么

| 路线 | 入口与适用范围 |
|---|---|
| 给温开放域 | `PlanarOpenSolver`，nx=1 均匀平面波；下壁给温且不透质量，上侧线性特征出口；压力直接从气体状态恢复 |
| 独立声脉冲 | 同一开放域求解器，测量入射/出射特征量与反射比 |
| 功率—膜温—气体耦合 | 旧闭域对称双侧保守耦合，尚未接入开放域 |
| Lim 近场对照 | 给温开放域对照精确声压；旧闭域使用实际热流换算声压，二者性质不同 |

已验证范围与数字见 [开放域报告](docs/OPEN_PLANAR_20260918.md)、
[Lim 热流对照](docs/LIM_NEARFIELD_20260918.md)。
当前开放域不包含有限宽膜边缘、三维衍射、直流温升和完整电热声器件响应。

## 环境与开始运行

在项目根目录使用现有 Python 3.14.2 虚拟环境：

```powershell
cd E:\CNT_thermophone_LBM
.\.venv\Scripts\Activate.ps1
# 查看参数展开和运行路径；不计算、不生成文件
python -m scripts.run_compute --config Compute/open_planar/study.yaml --batch 20260919_baseline --dry-run
# 先只运行一个短脉冲
python -m scripts.run_compute --config Compute/open_planar/study.yaml --batch 20260919_baseline --case pulse_r1
# 热声扫描为长计算；使用另一批次名，避免碰到已存在的算例
python -m scripts.run_compute --config Compute/open_planar/study.yaml --batch 20260919_scan
```

依赖固定在 [requirements-lock.txt](requirements-lock.txt)；重建环境时执行
`python -m venv .venv`，激活后运行 `python -m pip install -r requirements-lock.txt`。

## 如何新增算例

直接修改 [开放域配置示例](Compute/open_planar/study.yaml)。
`base_config` 相对于配置文件定位，指定公共物性和格子配置；`defaults` 是运行默认值，
`cases` 列出指定算例，`sweeps` 按参数列表的笛卡尔积生成扫描点：

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

生成名称如 `frequency__frequency_Hz_50000`；`--case` 可同时选择多个完整名称。
只改参数无需复制求解器或运行脚本。新物理或特殊驱动可通过短 Python 脚本调用公共接口。

- 所有输入采用 SI 单位；频率为温度/声学频率，不自动解释为电流频率。
- `refine` 同比缩小 dx、dt；域内实际长度和探针格点可能因取整略有差异。
- 默认在两周期内平滑启动，末两个单周期窗口检查稳定性；`analysis` 可配置拟合阶数、窗口和数值判据。
- 默认逐步保存探针与收支，最后两周期按约 128 点/周期保存剖面；由 `sampling` 配置。
- 脉冲默认相对密度幅值 1e-5、中心 0.3H、宽度 0.045H、探针 0.65H；
  完整 `pulse` 参数及其他默认值会写入有效配置。
- 运行前检查整个配置中的参数、名称和选择的目标路径。已有算例拒绝覆盖；新计算使用新批次。
- 顺序执行，遇到计算异常停止后续算例。进度在终端显示，不自动追加独立日志或报告。

## 后续直接调用哪些代码

| 要做的事 | 公共接口 | 注意事项 |
|---|---|---|
| 配置展开并运行一个算例 | [simulation](simulation/__init__.py)：`prepare_case`、`run_case` | 输出四个文件；不修改求解器物理算法 |
| 运行专题配置 | `simulation.run_study` | 指定批次，可选择算例，默认顺序执行 |
| 从已有数据重新分析 | `simulation.analyze_case` | 只读时序、配置和记录，不重新推进 LBM、不覆盖旧结果 |
| 自定义逐步给温 | [PlanarOpenSolver](core/planar_open_solver.py)：`step_temperature(T_K)`、`get_macro()` | 壁温在时间区间中点给定；压力在步末读取 |
| 周期气体计算 | [GasSolver2D](core/solver.py) | 与开放域的步进入口分开 |
| 薄膜功率耦合 | [run_levelc_predictor_corrector](coupling/conjugate.py)、[FilmOdeParams](coupling/film_ode.py) | 现有保守实现为闭域双侧耦合，不能直接代替开放域边界 |
| 幅相与周期诊断 | [diagnostics](postproc/diagnostics.py)：`fit_window`、`record`、`compare`；[fit_multiharmonic](postproc/multiharmonic_fit.py) | 区分总压均值、漂移和谐波；热流用区间中点 |
| 开放域时序分析 | [planar_open](postproc/planar_open.py)：`analyze_thermal`、`analyze_pulse` | 与时间推进独立，分析数据而非再跑仿真 |
| Lim 独立参照 | [lim2013](reference/lim2013.py)：`matched_gas`、`exact_pressure_hat`、`conductive_heat_flux_hat` | 用实际 LBM EOS 匹配参照；参照不进入热源或边界 |

最小 Python 例子（在项目根目录执行，输出路径必须尚不存在）：

```python
from pathlib import Path
from simulation import prepare_case, run_case, analyze_case

config = prepare_case({"name": "pulse_r1", "mode": "pulse", "refine": 1})
case_dir = Path("Compute/open_planar/python_example/raw/pulse_r1")
result = run_case(config, case_dir)
print(result["reflection_ratio"])

# 以后只做分析，无需重算
result = analyze_case(case_dir)
```

底层物理代码仍位于 `core/`、`boundary/`、`coupling/`；单位映射以
[unit_mapping](core/unit_mapping.py) 为准。`scripts/` 负责命令行，公共函数不再放在验证脚本中互相借用。

## 文件、状态和代码版本

```text
Compute/
  open_planar/
    study.yaml
    20260919_baseline/
      raw/<算例>/
        effective.yaml      # 全部物性、网格、驱动、采样、分析参数
        manifest.json       # 代码版本、环境、开始/结束时间、执行状态
        trajectory.npz      # 原始时序及按配置采样的剖面
        summary.json        # 数值结果与判据
      final/                # 按需生成图表
  _sources/                 # 未提交源码按实际版本共用一份归档
```

`manifest.status` 区分 `running/completed/failed/interrupted`；
`summary.validation_passed` 单独记录判据是否通过，未完成时为 null。
可捕获的异常/中断会在原文件中记录原因并保存已采样时序；这些时序不能作为完整稳态结果，
也不是可续算的分布函数检查点。进程被强制终止或断电时可能保留 running 状态，应人工确认。

代码及人工配置由 Git 管理；生成批次与源码归档不进入普通 Git 跟踪。
运行源码已固定在提交中时只记录提交号和源码哈希；存在未提交或新增源码时，在
`Compute/_sources/` 保存一份共享快照，manifest 优先使用相对路径引用（跨盘时使用绝对路径）。不会自动提交项目代码，
不会解压源码复跑。历史快照继续保留在原批次的 `raw/` 中。

正常计算不生成 README、排版检查或 `_work/`。专项绘图独立运行：

```powershell
python -m scripts.plot_planar_open --input Compute/open_planar/20260918 --output Compute/open_planar/20260918 --overwrite
```

这是已有完整验证组的三面板图，需要 50 kHz、约 1 mm 域长、r=1/2/4 热声组及 r=1/4 脉冲组；
不是任意单算例绘图器。默认导出 PNG、PDF、汇总 CSV 及必要的绘图源数据；
只有显式传入 `--qa-tools` 才输出排版检查，SVG 用 `--formats png pdf svg` 选择。

## 已有结果与其他入口

- [开放域图表](Compute/open_planar/20260918/final/) / [原始数据](Compute/open_planar/20260918/raw/)
- [Lim 热流对照图表](Compute/lim_nearfield/20260918_r1_r2/final/) / [四倍加密数据](Compute/lim_nearfield/20260918_r4/raw/)
- [历史 Lim 入口结果](Compute/lim2013/)：保留当时参数口径，不作为新的精度声明。

历史 `raw/final` 已迁入 `Compute/`，字节内容保留；旧 `_work` 仍在 `results/` 待以后单独处理。
历史 manifest 中的启动路径保留为当时的记录。迁移不生成新的一套说明文件。

旧的 `verify_planar_open`、`verify_lim_nearfield`、`verify_lim2013` 和
`verify_nearfilm_chain` 命令保留，见 [脚本入口](scripts/README.md)。
本轮四文件契约用于新的开放域工作流；旧闭域入口继续保留其原有数据格式。

回归入口为 `python -m pytest -q`。新增或修改公共代码时运行对应回归；普通参数扫描只做
算例自身的数值检查。链路约定见 [CHAIN](docs/CHAIN.md)，迁移溯源见 [PROVENANCE](docs/PROVENANCE.md)。
