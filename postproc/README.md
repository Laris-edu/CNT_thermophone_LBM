# postproc/

| 文件 | 用途 |
|---|---|
| [multiharmonic_fit.py](multiharmonic_fit.py) | 趋势与多谐波联合拟合、稳定窗与符号工具；复声压评估用，不继承旧协议表 |
| [diagnostics.py](diagnostics.py) | 时间窗拟合、复幅值记录、参照幅相误差与稳定性比较 |
| [planar_open.py](planar_open.py) | 从已有数组分析开放热声/脉冲，生成结果与判据，不推进 LBM |

探头读数是总压；周期声压必须先用本模块分离均值、热漂移与目标谐波。旧的单音 `harmonic_fit.py` 未迁入（有 DC/趋势时应使用联合拟合器）。
