"""motion_bvh — 带运动界（motion bounds）的 BVH 射线遍历可复用题库。

仅依赖 Python 3.14 标准库。典型用法::

    from motion_bvh import build, Ray

    bvh = build([(vertices0, vertices1), ...])   # 每个三角形给出 t=0/1 两端顶点
    hit = bvh.intersect(Ray(origin, direction), time=0.5)
    if hit is not None:
        print(hit.face_id, hit.t, hit.barycentric)

公开契约见各对象文档字符串与仓库 README。
"""

from .core import (
    BIN_COUNT,
    LEAF_MAX,
    MAX_TRIANGLES,
    BudgetExhaustedError,
    HitResult,
    MotionBVH,
    MotionBVHError,
    Ray,
    TraceStats,
    Triangle,
    build,
    intersect,
    intersect_full,
)

__all__ = [
    "MAX_TRIANGLES",
    "LEAF_MAX",
    "BIN_COUNT",
    "Triangle",
    "Ray",
    "HitResult",
    "TraceStats",
    "MotionBVH",
    "MotionBVHError",
    "BudgetExhaustedError",
    "build",
    "intersect",
    "intersect_full",
]

__version__ = "1.0.0"
