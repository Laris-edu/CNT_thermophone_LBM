# boundary/

热壁与平面开放出口。旧共轭链路用对称 mass-neutral；单侧给温验证走独立入口。

| 文件 | 用途 |
|---|---|
| [wall_thermal_mass_neutral.py](wall_thermal_mass_neutral.py) | 对称双侧热带：按新膜温重建壁行并保持密度，链路主用；实际区间能量即交换量 |
| [wall_thermal_grad.py](wall_thermal_grad.py) | Grad 型非平衡外推热壁，A/B 夹具与耦合初始化导入 |
| [wall_dirichlet.py](wall_dirichlet.py) | 给温平衡钳制壁；旧 Heun 路径与初始化的兼容入口 |
| [wall_common.py](wall_common.py) | 方向与多速链接工具 |
| [planar_open.py](planar_open.py) | 全容纳漫反射给温壁、线性正入射特征出口、跨边界真实布居收支；覆盖 D2Q37 全三层链接 |

未迁入：`open_cbc.py`、`open_sponge.py`（开放边界，未认证）、`wall_neumann.py`、`wall_mass_audit.py`（旧夹具与审计）。row1 传导矩只作诊断，不能当交换量使用。

新 `planar_open.py` 不复用旧仓开放边界；其特征出口仅面向静止背景、小振幅、
正入射平面声波，不能宣称为完整多维热 CBC。壁温是漫反射分布的温度，
首格气体位于 z=dx/2，不能将其温度强制等同壁温。误差与独立反射测量见
[开放域报告](../docs/OPEN_PLANAR_20260918.md)。

开放膜耦合每步建立一个 `PreparedDiffuseWall`，复用碰撞后的出射布居。
`energy_exchange_lu(T)` 从试探壁温对应的实际入射布居计算区间交换量，
不写入流步状态；`write(T, f_new, g_new)` 只在膜温确定后填入所有入射链接。
对象只供当前时间步使用，不跨步缓存气体状态。最终审计仍独立读取写入后的跨界布居，
`diffuse_temperature_wall(...)` 给温入口保持兼容。
