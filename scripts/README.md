# scripts/

| 文件 | 用途 |
|---|---|
| [run_gas_smoke.py](run_gas_smoke.py) | 周期气体启动检查：小扰动、有限步、质量/能量守恒与温度演化 |
| [verify_nearfilm_chain.py](verify_nearfilm_chain.py) | 近膜链路 A/B/C 驱动：按模式给定壁温/单侧热流/面功率，输出时间序列、独立 NSF 对照与审计 |
| [verify_lim2013.py](verify_lim2013.py) | Lim 2013 近/远场复现：解析曲线（Fig. 2/4/5）+ LBM 壁面热流换算声压的对照 |
| [verify_kirchhoff.py](verify_kirchhoff.py) | 2D Kirchhoff 核的制造解验证（对应 `verification/test_phase4_kirchhoff.py`） |

```powershell
python -m scripts.run_gas_smoke
python -m scripts.verify_nearfilm_chain --mode C --output results\case_C   # 3 周期约 4 万步，长
python -m scripts.verify_lim2013 --mode both --output results\lim2013      # 解析 + LBM，约 2 分钟
```

`verify_lim2013` 的声压一律由壁面热流经 `p_hat = (gamma-1) q_hat / c0` 换算，**不读压力探针**：
本仓是封闭周期域，探针读等容压缩，在 10 kHz 工作点比辐射声压大 93 倍（实测盒压 35.12 Pa
与准静态式 `2(gamma-1)q/(Omega H)` 完全一致）。

`verify_nearfilm_chain` 的完整三周期算例是长时运行：请显式启动，不要挂在后台无人看守。未迁入：`verify_local_transport.py`（局部输运矩阵扫描）、`check_environment.py`（旧仓环境检查）。
