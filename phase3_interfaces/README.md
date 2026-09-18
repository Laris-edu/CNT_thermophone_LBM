# phase3_interfaces/

热流与复幅相的对外接口。

| 文件 | 用途 |
|---|---|
| [heat_flux_extraction.py](heat_flux_extraction.py) | 壁面传导热流提取与 LU→SI 转换；显式传当前配置时接 local 热流系数，测的是传导矩 |
| [wall_state_contract.py](wall_state_contract.py) | 由权威 mapping 转换壁温单位；法向约定限当前上半域 |
| [complex_amplitude.py](complex_amplitude.py) | 幅相、模态与衰减工具；保持复指数与时间零点约定 |

未迁入：`modal_fit.py`、`probe_sampling.py`、`run_hdf5.py`（旧阶段的后处理与交接入口；HDF5 输出现由 `core/solver.py` 的 `save_hdf5` 直接提供）。
