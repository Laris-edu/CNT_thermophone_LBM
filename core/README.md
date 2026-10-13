# core/

气体核：D2Q37 多速格子 + f-g 双分布（D=2, S=3）+ 局部输运。只保留本项目实际运行的一条配置。

配置必须显式指定 `transport.mode: local`；frozen 和手动固定热流系数已退役。
参考态 τ/热流系数保留在元数据中，运行时统一使用逐格点系数。

| 文件 | 用途 |
|---|---|
| [lattice.py](lattice.py)、[lattice_d2q37.py](lattice_d2q37.py) | 速度集注册与 D2Q37 权重/奇偶矩；D2Q21 已退役，构造即报错 |
| [equilibrium.py](equilibrium.py) | 四阶 Hermite f 平衡态、二阶 g 平衡态；逐布居正性检查，不裁剪 |
| [collision_smrt.py](collision_smrt.py) | 二阶中心矩闭合、measured 偏应力、tau22 迹通道、热流最小范数投影；逐格守恒 |
| [collision_numba.py](collision_numba.py) | 可选的串行 CPU 编译碰撞内核；复用公共宏观量、平衡态、局部输运与矩投影 |
| [streaming.py](streaming.py) | 周期 pull 流步及独立 `pull_stream_open_y`（越界链接以 NaN 哨兵显式待填） |
| [macroscopic.py](macroscopic.py) | 宏观量恢复、能量闭合定义与传导热流读出 |
| [unit_mapping.py](unit_mapping.py) | 单位与输运映射的唯一权威；旧配置的退役旋钮会被显式拒绝 |
| [solver.py](solver.py) | `GasSolver2D`：collide → stream → boundary 三步链 |
| [planar_open_solver.py](planar_open_solver.py) | `PlanarOpenSolver`：nx=1 单侧给温入口，法向不回绕；独立记录壁/出口质量与能量交换 |

开放平面验证使用 `step_temperature(T_midpoint_K)`，不调用旧膜 ODE。
当前边界与适用范围见 [开放域报告](../docs/OPEN_PLANAR_20260918.md)。
碰撞的零低阶矩投影缓存固定 Gram 逆映射；这是中心矩平移的代数等价加速，
并未将流速设为零，有限流速等价性由 `test_collision_projection.py` 守护。

基础配置中的 `numerics.collision_backend` 可显式设为 `numba`；默认 `numpy`。
`collide_fg(..., backend='numba')`、周期求解器和开放域求解器使用同一入口。
Numba 路线将逐格碰撞、中心矩和守恒修正融合为机器码，单位映射、输运本构、
平衡态检查、边界和膜温求根继续使用现有公共实现。它使用 float64、串行循环，
关闭 `fastmath`，不裁剪布居，不自动回退到 NumPy。
依赖由 requirements 文件锁定；首次调用会编译或加载缓存，后续调用复用内核。
缓存位于 Python 的 `__pycache__` 中，不属于算例结果。
开放域 `manifest.json` 和周期求解器 HDF5 元数据记录所选后端，以及 Numba 路线的 Numba/llvmlite 版本。
短时对照可运行 `python -m Compute.open_planar.benchmark_cpu --backend numba
--output <新路径.npz> --compare <numpy基准.npz>`；计时区分预热与后续完整耦合步骤。

有限条膜的 `FiniteStripSolver` 和 `simulation.run_strip_case` 支持显式
`numerics.collision_backend: cupy`；专题参数使用 `collision_backend: cupy`。
安装可选依赖用 `python -m pip install -r requirements-gpu.txt`，需要兼容的 NVIDIA
驱动及 CUDA 12.x 工具包；用 `CUDA_VISIBLE_DEVICES` 选择单张 GPU。
`collision_gpu.py`、`finite_strip_gpu.py` 使用 float64 CUDA 内核，关闭快速数学和 FMA
收缩；输运仍调用 `unit_mapping.local_transport_coefficients`，保持实测偏应力、tau22
与逐格守恒。布居、宏观场和 PML 状态留在 GPU，膜温求根沿用 CPU 公共实现，只交换
膜上数据；采样下载观察点，四文件输出时才下载末态场。缺少 GPU 或 CuPy 直接报错，
不自动回退。`manifest.json` 记录实际设备、精度、CuPy/CUDA 版本及显存。
此后端目前覆盖有限条膜四文件入口，不包含平面/周期求解器、HDF5 或热流矩诊断接口。
CPU/GPU 等价检查不代替算例的边界、源场和传播验证。

迁移时删除：`lattice_d2q21.py`、`polyatomic_fg.py`、`retired_spectral.py`、`hermite.py`（其中被使用的 `monomial_exponents` 已内联进 `equilibrium.py`），以及 `collision_smrt.py`／`unit_mapping.py`／`solver.py` 内的 ghost 迹策略族、四阶闭合、解析应变率偏应力、谱投影与退役配置保护。逐条来历见 [docs/PROVENANCE.md](../docs/PROVENANCE.md)。
