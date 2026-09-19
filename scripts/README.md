# scripts/

常规开放域算例从 [根 README](../README.md) 和 [study.yaml](../Compute/open_planar/study.yaml) 开始。
命令行只处理参数，运行、拟合、理论参照分别调用公共模块。

| 入口 | 用途 |
|---|---|
| [run_compute.py](run_compute.py) | 推荐入口：配置展开、单算例或参数扫描，首轮顺序执行 |
| [verify_planar_open.py](verify_planar_open.py) | 兼容旧参数；通过同一个公共运行接口执行给温/脉冲算例 |
| [plot_planar_open.py](plot_planar_open.py) | 专项三面板图，需要 50 kHz、1 mm、r=1/2/4 热声组及 r=1/4 脉冲组 |
| [verify_lim_nearfield.py](verify_lim_nearfield.py) | 原闭域近场专项：匹配实际 EOS，比较膜温、热流和 Lim 式 (22) |
| [plot_lim_nearfield.py](plot_lim_nearfield.py) | 从已有闭域专项结果绘制复幅误差图并导出 CSV |
| [verify_lim2013.py](verify_lim2013.py) | 保留旧解析曲线与 LBM 壁面热流换算声压入口 |
| [verify_nearfilm_chain.py](verify_nearfilm_chain.py) | 原近膜 A/B/C 驱动：给壁温、单侧热流或面功率 |
| [run_gas_smoke.py](run_gas_smoke.py) | 周期气体小扰动及守恒检查 |
| [verify_kirchhoff.py](verify_kirchhoff.py) | 2D Kirchhoff 核制造解验证 |
| [output_layout.py](output_layout.py) | 兼容导入；实际输出工具已移入 simulation |

## 配置运行和旧开放域命令

```powershell
python -m scripts.run_compute --config Compute/open_planar/study.yaml --batch new_batch --dry-run
python -m scripts.run_compute --config Compute/open_planar/study.yaml --batch new_batch --case pulse_r1

# 原命令继续可用；选一个尚不存在的算例目录
python -m scripts.verify_planar_open --mode pulse --refine 4 --output Compute/open_planar/legacy_entry --case pulse_r4
python -m scripts.verify_planar_open --mode thermal --frequency 50000 --refine 4 --output Compute/open_planar/legacy_entry --case f50000_r4
```

开放域默认给定 `300+0.1*r(t)*sin(2*pi*f*t)` K，频率指温度/声学频率。
默认 refine=4、1 mm 域长、6 周期、2 周期平滑启动；细网格热声是长计算。
压力从气体 rho*theta 恢复，热流只作独立诊断。

两种命令共用 `simulation.run_case`，每算例仅有 effective.yaml、manifest.json、
trajectory.npz、summary.json。旧入口省略 `--case` 时直接写入批次 `raw/`；
指定后写入 `raw/<case>/`。已存在目录拒绝覆盖。
未提交源码在项目 `Compute/_sources/` 按版本共用快照，不在每个算例下复制源码。

公共函数从 `simulation`、`postproc.diagnostics`、`reference.lim2013` 导入，
不再从另一个运行脚本导入。Python 调用例子和适用范围见根 README。

## 独立绘图

```powershell
python -m scripts.plot_planar_open --input Compute/open_planar/20260918 --output Compute/open_planar/20260918 --overwrite
python -m scripts.plot_lim_nearfield --input Compute/lim_nearfield/20260918_r1_r2 Compute/lim_nearfield/20260918_r4 --output Compute/lim_nearfield/20260918_r1_r2 --overwrite
```

输入可以是批次或 raw 目录；Lim 绘图还可直接接收 summary.json。
默认 PNG、PDF、汇总 CSV 写入 `final/`；开放域绘图数据写入 `raw/plot_data/`。
SVG 用 `--formats png pdf svg` 选择。仅显式传入 `--qa-tools <排版检查工具目录>`
才生成 `_work/qa/`。重画已有图表需 `--overwrite`，不重新推进求解器。

## 保留的闭域入口

```powershell
python -m scripts.run_gas_smoke
python -m scripts.verify_nearfilm_chain --mode C --output Compute/nearfilm/new_batch
python -m scripts.verify_lim2013 --mode both --output Compute/lim2013/new_batch
python -m scripts.verify_lim_nearfield --output Compute/lim_nearfield/new_batch --frequencies 50000 75000 100000 --refinements 1 2 4
```

这些入口保留原有输出格式，正式数据同样放入批次 `raw/`；本轮未统一其内部数据结构。
`verify_nearfilm_chain` 完整三周期约四万步，属于长计算。

`verify_lim2013` 的声压由壁面热流经 `p_hat=(gamma-1)*q_hat/c0` 换算，
不作为直接压力验证。它采用封闭周期域，探针含盒内压缩；
默认 10 kHz、ny=64 下，盒压峰值/辐射峰值约为 66 倍。
旧记录中的 93 倍混用了峰值和 RMS，现行诊断已修正；Lim 参照匹配实际 LBM EOS 压强。

`verify_lim_nearfield` 使用零均值 AC 驱动、单侧实际交换热流换算及固定物理域长的 dx/dt 加密。
它同时检查膜温，不能仅凭热流吻合判定全链路通过。科学边界和历史数据见
[近场对照报告](../docs/LIM_NEARFIELD_20260918.md)、
[开放域报告](../docs/OPEN_PLANAR_20260918.md)。

旧仓 `verify_local_transport.py` 与 `check_environment.py` 未迁入本项目。
