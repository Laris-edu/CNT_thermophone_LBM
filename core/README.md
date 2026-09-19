# core/

气体核：D2Q37 多速格子 + f-g 双分布（D=2, S=3）+ 局部输运。只保留本项目实际运行的一条配置。

| 文件 | 用途 |
|---|---|
| [lattice.py](lattice.py)、[lattice_d2q37.py](lattice_d2q37.py) | 速度集注册与 D2Q37 权重/奇偶矩；D2Q21 已退役，构造即报错 |
| [equilibrium.py](equilibrium.py) | 四阶 Hermite f 平衡态、二阶 g 平衡态；逐布居正性检查，不裁剪 |
| [collision_smrt.py](collision_smrt.py) | 二阶中心矩闭合、measured 偏应力、tau22 迹通道、热流最小范数投影；逐格守恒 |
| [streaming.py](streaming.py) | 周期 pull 流步及独立 `pull_stream_open_y`（越界链接以 NaN 哨兵显式待填） |
| [macroscopic.py](macroscopic.py) | 宏观量恢复、能量闭合定义与传导热流读出 |
| [unit_mapping.py](unit_mapping.py) | 单位与输运映射的唯一权威；旧配置的退役旋钮会被显式拒绝 |
| [solver.py](solver.py) | `GasSolver2D`：collide → stream → boundary 三步链 |
| [planar_open_solver.py](planar_open_solver.py) | `PlanarOpenSolver`：nx=1 单侧给温入口，法向不回绕；独立记录壁/出口质量与能量交换 |

开放平面验证使用 `step_temperature(T_midpoint_K)`，不调用旧膜 ODE。
当前边界与适用范围见 [开放域报告](../docs/OPEN_PLANAR_20260918.md)。
碰撞的零低阶矩投影缓存固定 Gram 逆映射；这是中心矩平移的代数等价加速，
并未将流速设为零，有限流速等价性由 `test_collision_projection.py` 守护。

迁移时删除：`lattice_d2q21.py`、`polyatomic_fg.py`、`retired_spectral.py`、`hermite.py`（其中被使用的 `monomial_exponents` 已内联进 `equilibrium.py`），以及 `collision_smrt.py`／`unit_mapping.py`／`solver.py` 内的 ghost 迹策略族、四阶闭合、解析应变率偏应力、谱投影与退役配置保护。逐条来历见 [docs/PROVENANCE.md](../docs/PROVENANCE.md)。
