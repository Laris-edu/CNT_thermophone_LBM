"""速度集注册：本项目只保留 D2Q37。

D2Q21 在 Phase 2 迁移中退役：四阶平衡态需要八阶求积，D2Q21 不满足。
旧代码中``velocity_set`` 缺省即为 D2Q21，会让配置缺项静默退化到不能运行的
速度集；这里改为 D2Q37 为唯一取值，并让 D2Q21 立即报错。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from core.lattice_d2q37 import THETA_Q_D2Q37, LatticeD2Q37, make_d2q37


class Lattice(Protocol):
    c: np.ndarray
    w: np.ndarray
    theta_q: float
    opposite: np.ndarray
    q: int
    d: int


LatticeFamily = LatticeD2Q37

CANONICAL_NAME = "D2Q37"

_RETIRED = {
    "D2Q21": (
        "D2Q21 已退役：四阶平衡态需要八阶求积，本项目使用 D2Q37。"
        "历史结果请用旧仓的对应 commit 复现。"
    ),
}


def normalize_velocity_set(value: str | None) -> str:
    """返回唯一支持的速度集名；``None`` 表示未指定，按 D2Q37 处理。"""

    if value is None:
        return CANONICAL_NAME
    name = str(value).strip().upper()
    if name in _RETIRED:
        raise ValueError(_RETIRED[name])
    if name != CANONICAL_NAME:
        raise ValueError(f"unsupported velocity_set: {value}")
    return CANONICAL_NAME


def make_lattice(velocity_set: str | None = None, dtype=np.float64) -> LatticeD2Q37:
    normalize_velocity_set(velocity_set)
    return make_d2q37(dtype=dtype)


def default_theta_q(velocity_set: str | None = None) -> float:
    normalize_velocity_set(velocity_set)
    return THETA_Q_D2Q37


def default_q(velocity_set: str | None = None) -> int:
    normalize_velocity_set(velocity_set)
    return 37
