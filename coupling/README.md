# coupling/

功率 → 膜温环节。沿 x 均匀的平面薄膜与二维 D2Q37 气体核耦合，`nx` 列数不代表有限宽膜的几何边缘。封闭双侧接口与开放上半域接口分别列在下表；开放接口使用实际壁面交换联立膜储能，仅用于当前的小振幅正入射局部平面设置。新工况的网格、域长和出口适用性需单独检查。

| 文件 | 用途 |
|---|---|
| [conjugate.py](conjugate.py) | 链路入口 `run_levelc_predictor_corrector`；默认 `scheme="conservative"` |
| [conservative.py](conservative.py) | 热壁重构与膜储能在同一步联立；只支持封闭对称双侧（`gas_flux_factor=2`） |
| [film_ode.py](film_ode.py) | `C_A·dT_s/dt = P_in − 2q″_g` 与 `FilmOdeParams` |
| [drive.py](drive.py) | SI 面功率驱动信号（正弦/阶跃/高斯），允许有符号功率 |
| [energy_audit.py](energy_audit.py) | 区间能量审计结果类型与容差比较 |
| [planar_open.py](planar_open.py) | 开放上半域与对称双侧单温膜的中点保守耦合；使用实际壁面交换，返回成功前不更新膜状态 |

接口细节与限制见根 [README](../README.md)。
