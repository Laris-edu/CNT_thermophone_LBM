# coupling/

功率 → 膜温环节。当前为沿 x 均匀的一维平面薄膜设置：二维 D2Q37 气体核上施加整行膜温，x 周期延拓，沿 x 平均的热流反馈给单温膜 ODE；`nx` 列数不代表有限宽膜的几何边缘。法向边界的开放性与声压准确性未建立。

| 文件 | 用途 |
|---|---|
| [conjugate.py](conjugate.py) | 链路入口 `run_levelc_predictor_corrector`；默认 `scheme="conservative"` |
| [conservative.py](conservative.py) | 热壁重构与膜储能在同一步联立；只支持封闭对称双侧（`gas_flux_factor=2`） |
| [film_ode.py](film_ode.py) | `C_A·dT_s/dt = P_in − 2q″_g` 与 `FilmOdeParams` |
| [drive.py](drive.py) | SI 面功率驱动信号（正弦/阶跃/高斯），允许有符号功率 |
| [energy_audit.py](energy_audit.py) | 区间能量审计结果类型与容差比较 |

接口细节与限制见根 [README](../README.md)。
