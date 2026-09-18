# verification/

链路回归与四个动态测量器。`pytest -q` 全绿是裁剪或改动后的判据；阈值只在测试条件内成立，不是器件误差界限。

| 文件 | 守护什么 |
|---|---|
| [test_conservative_chain.py](test_conservative_chain.py) | 膜气能量闭合、膜温与壁温同值、零输入不假失败、非保守组合被拒、声压源项能量极限 |
| [test_local_handoff.py](test_local_handoff.py) | 非零传导热流的提取与 LU/SI 转换一致 |
| [test_local_transport.py](test_local_transport.py) | 局部物性本构、均匀/非均匀态的输运系数与 NSF 对照 |
| [test_local_transport_dynamics.py](test_local_transport_dynamics.py) | 热稀薄运动气体、有限温差加密、声衰减保持 |
| [test_quadrature_equilibrium.py](test_quadrature_equilibrium.py) | 平衡态的独立张量展开、矩、正性；退役 D2Q21 立即报错 |
| [test_analytic_stress_transport.py](test_analytic_stress_transport.py) | 声速/衰减、剪切黏性、热扩散、零梯度下的应力松弛 |
| [test_collision_sign_and_conservation.py](test_collision_sign_and_conservation.py) | 实测偏应力的 Navier–Stokes 符号、非平凡扰动下的逐格守恒 |
| [test_multiharmonic_fit.py](test_multiharmonic_fit.py) | 联合谐波拟合器在合成信号上的正确性 |
| [test_phase2_d2q37_fallback.py](test_phase2_d2q37_fallback.py) | D2Q37 静态矩与对偶速度 |
| [test_phase3_film_ode.py](test_phase3_film_ode.py) | 薄膜 ODE 制造解与幅相 |
| [test_lim2013.py](test_lim2013.py) | Lim 2013 精确解自洽、Eq. (22) 的适用频段与 1 MHz 失效、紧致热层恒等式、Rayleigh 距离复现论文数字、本仓链路与 Eq. (22) 的比值等于闭式 EOS 常数 |
| [test_phase4_kirchhoff.py](test_phase4_kirchhoff.py)、[test_phase4_d3_compact_source.py](test_phase4_d3_compact_source.py) | 远场核与紧致源的制造解（远场本身未认证） |
| [test_phase1_*.py](test_phase1_constants.py) | 所选参考工具的代数与相位回归 |
| [shear_wave_measurement.py](shear_wave_measurement.py)、[thermal_diffusion_measurement.py](thermal_diffusion_measurement.py)、[acoustic_wave_measurement.py](acoustic_wave_measurement.py)、[local_transport_measurement.py](local_transport_measurement.py) | 剪切黏性、热扩散、声速/衰减、局部输运的动态测量器（被上面的测试调用） |

未迁入：`test_phase2_postprocess_modal_fit.py`（随 `modal_fit` 移除）。
