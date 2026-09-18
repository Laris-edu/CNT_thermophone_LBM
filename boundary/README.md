# boundary/

热壁。当前链路只用对称 mass-neutral；其余两个用于 A/B 夹具与初始化。

| 文件 | 用途 |
|---|---|
| [wall_thermal_mass_neutral.py](wall_thermal_mass_neutral.py) | 对称双侧热带：按新膜温重建壁行并保持密度，链路主用；实际区间能量即交换量 |
| [wall_thermal_grad.py](wall_thermal_grad.py) | Grad 型非平衡外推热壁，A/B 夹具与耦合初始化导入 |
| [wall_dirichlet.py](wall_dirichlet.py) | 给温平衡钳制壁；旧 Heun 路径与初始化的兼容入口 |
| [wall_common.py](wall_common.py) | 方向与多速链接工具 |

未迁入：`open_cbc.py`、`open_sponge.py`（开放边界，未认证）、`wall_neumann.py`、`wall_mass_audit.py`（旧夹具与审计）。row1 传导矩只作诊断，不能当交换量使用。
