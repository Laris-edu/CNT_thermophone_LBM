# farfield/

二维频域远场外推。**未认证**：目前只有制造解级验证，且不在当前近膜链路内；开放器件的声压结论不能由它直接给出。

| 文件 | 用途 |
|---|---|
| [kirchhoff_2d.py](kirchhoff_2d.py) | 2D Kirchhoff/Hankel 频域外推核，配合 [scripts/verify_kirchhoff.py](../scripts/verify_kirchhoff.py) 的制造解夹具 |
| [compact_source.py](compact_source.py) | 常物性紧致源模型 |

对应回归：`verification/test_phase4_kirchhoff.py`、`verification/test_phase4_d3_compact_source.py`。
