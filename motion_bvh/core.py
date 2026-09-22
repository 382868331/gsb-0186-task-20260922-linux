"""motion_bvh — 带运动界的 BVH 射线遍历（可复用题库）。

公开接口见包 ``__init__``；本模块只使用 Python 标准库。

数学约定
========
* 每个三角形提供 t=0 与 t=1 两组顶点，顶点位置在时间区间 [0, 1] 内线性插值。
* BVH 每个节点保存两端时刻的 AABB，查询时按同一时间线性插值。
* 坐标与 t 均按 IEEE-754 binary64（Python float）计算。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# 公开常量
# ---------------------------------------------------------------------------

MAX_TRIANGLES = 300
"""build() 接受的三角形数量上限（接口边界）。"""

LEAF_MAX = 4
"""叶节点最多容纳的三角形数。"""

BIN_COUNT = 8
"""SAH 分桶使用的固定桶数。"""

GUARD = float.fromhex("0x1.0p-48")
"""
浮点舍入护栏（相对量，约 3.55e-15；O(1) 坐标上约十几个 ULP）。

仅用于吸收“同一数学量经由不同运算路径”产生的舍入：
  * 查询时刻的 AABB 在插值后按 GUARD 向外微扩，保证插值三角形不被错误剔除；
  * barycentric 边值与 t_min/t_max 闭区间判定允许 GUARD 级越界。
它不是用来放宽几何契约的宽限差。
"""

TIE_EPS = float.fromhex("0x1.0p-45")
"""
同距判定带（相对量，约 2.84e-14）。
两次命中的射线参数之差落在该带内视为“同距”，按 face_id 取小。
"""


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------


class MotionBVHError(ValueError):
    """非法输入错误。消息携带可定位的字段路径（如 ``triangles[3].vertices1[2][1]``）。"""


class BudgetExhaustedError(RuntimeError):
    """遍历搜索预算耗尽。

    与“证明无命中”（``intersect`` 返回 ``None``）严格区分：抛出本异常表示
    未能完成证明，调用方不得把它当作 miss。
    """

    def __init__(self, message: str, nodes_visited: int, triangle_tests: int) -> None:
        super().__init__(message)
        self.nodes_visited = nodes_visited
        self.triangle_tests = triangle_tests


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Triangle:
    """一个三角形在两端时刻的顶点；时间区间 [0, 1] 内逐顶点线性插值。

    ``vertices0`` / ``vertices1`` 各为 3 个三维点 (a, b, c)；静态三角形把
    两端填成相同值即可。顶点顺序决定 barycentric 的 (u, v, w) 对应关系
    （分别对应 a, b, c）。零面积三角形允许构建，但任何时刻都不会被命中。
    """

    vertices0: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    vertices1: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

    def __post_init__(self) -> None:
        v0 = _tri_vertices(self.vertices0, "triangle.vertices0")
        v1 = _tri_vertices(self.vertices1, "triangle.vertices1")
        object.__setattr__(self, "vertices0", v0)
        object.__setattr__(self, "vertices1", v1)


@dataclass(frozen=True)
class Ray:
    """射线 origin + t*direction。

    * ``direction`` 不要求归一化（t 为世界单位距离）；允许分量为 0，但不许全零。
    * ``tmin`` 必须有限；``tmax`` 允许有限数或 +inf（-inf/NaN 拒绝）。
    * 闭区间 [tmin, tmax] 上的命中都有效。
    """

    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    tmin: float = 0.0
    tmax: float = float("inf")

    def __post_init__(self) -> None:
        origin = _vec3(self.origin, "ray.origin")
        direction = _vec3(self.direction, "ray.direction")
        if not any(component != 0.0 for component in direction):
            raise MotionBVHError(
                "ray.direction: must be a non-zero vector (zero components are allowed)"
            )
        tmin = _number(self.tmin, "ray.tmin")
        tmax = _ray_bound(self.tmax, "ray.tmax")
        if tmin > tmax:
            raise MotionBVHError(f"ray.tmin ({tmin}) must be <= ray.tmax ({tmax})")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "tmin", tmin)
        object.__setattr__(self, "tmax", tmax)


@dataclass(frozen=True)
class HitResult:
    """最近命中结果。"""

    face_id: int
    """被命中三角形在 build 输入中的序号（0 起）。"""

    t: float
    """命中处的射线参数（世界单位）；结果夹紧到 [ray.tmin, ray.tmax]。"""

    barycentric: tuple[float, float, float]
    """(u, v, w)，分别对应该面顶点 a, b, c，归一化后之和为 1。"""

    triangle_tests: int
    """本次遍历实际执行的射线-三角形测试次数（含退化面测试）。"""

    nodes_visited: int
    """本次遍历弹出并做了 AABB 测试的 BVH 节点数（含根节点与被剔除节点）。"""


@dataclass(frozen=True)
class TraceStats:
    """一次遍历的访问计数（命中与证明无命中时均可取得）。"""

    triangle_tests: int
    """实际执行的射线-三角形测试次数（含退化面测试）。"""

    nodes_visited: int
    """弹出并做了 AABB 测试的 BVH 节点数（含根节点与被剔除节点）。"""


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _number(value: object, where: str) -> float:
    """接受 int/float（拒绝 bool），拒绝 NaN/Infinity。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MotionBVHError(
            f"{where}: expected a finite real number, got {type(value).__name__}"
        )
    result = float(value)
    if math.isnan(result) or math.isinf(result):
        raise MotionBVHError(f"{where}: NaN/Infinity is not allowed")
    return result


def _ray_bound(value: object, where: str) -> float:
    """ray 区间界专用：有限数（可为负）或 +inf；拒绝 NaN/-inf/bool。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MotionBVHError(
            f"{where}: expected a real number or +inf, got {type(value).__name__}"
        )
    result = float(value)
    if math.isnan(result):
        raise MotionBVHError(f"{where}: NaN is not allowed")
    if math.isinf(result) and result < 0.0:
        raise MotionBVHError(f"{where}: -inf is not allowed")
    return result


def _vec3(value: object, where: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise MotionBVHError(f"{where}: expected a 3-component sequence (x, y, z)")
    return tuple(  # type: ignore[return-value]
        _number(component, f"{where}[{k}]") for k, component in enumerate(value)
    )


def _tri_vertices(value: object, where: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise MotionBVHError(f"{where}: expected 3 vertices")
    return tuple(  # type: ignore[return-value]
        _vec3(point, f"{where}[{k}]") for k, point in enumerate(value)
    )


def _coerce_triangle(value: object, index: int) -> tuple:
    if isinstance(value, Triangle):
        return (value.vertices0, value.vertices1)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        v0 = _tri_vertices(value[0], f"triangles[{index}].vertices0")
        v1 = _tri_vertices(value[1], f"triangles[{index}].vertices1")
        return (v0, v1)
    raise MotionBVHError(
        f"triangles[{index}]: expected Triangle or (vertices0, vertices1), "
        f"got {type(value).__name__}"
    )


def _query_time(value: object) -> float:
    time = _number(value, "time")
    if not 0.0 <= time <= 1.0:
        raise MotionBVHError(
            f"time: expected a value in [0, 1] (vertices are defined on t in 0..1), got {time}"
        )
    return time


def _budget(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MotionBVHError(
            f"budget: expected a positive int or None, got {type(value).__name__}"
        )
    if value <= 0:
        raise MotionBVHError(f"budget: expected a positive int, got {value}")
    return value


# ---------------------------------------------------------------------------
# 向量小工具（元组运算，无外部库）
# ---------------------------------------------------------------------------


def _sub(p: tuple[float, ...], q: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(p[k] - q[k] for k in range(3))


def _cross(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# ---------------------------------------------------------------------------
# BVH 节点与构建
# ---------------------------------------------------------------------------


class _Node:
    __slots__ = ("b0", "b1", "left", "right", "ids", "axis")

    def __init__(
        self,
        b0: tuple,
        b1: tuple,
        left: Optional["_Node"],
        right: Optional["_Node"],
        ids: Optional[tuple[int, ...]],
        axis: int,
    ) -> None:
        # 端点包围盒：((lox,loy,loz),(hix,hiy,hiz))
        self.b0 = b0
        self.b1 = b1
        self.left = left
        self.right = right
        self.ids = ids  # 非 None 即叶节点，且按 face_id 升序
        self.axis = axis


def _tri_bounds(verts: tuple) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        tuple(min(v[k] for v in verts) for k in range(3)),
        tuple(max(v[k] for v in verts) for k in range(3)),
    )


def _union_bounds(bounds: Sequence[tuple]) -> tuple:
    los = [b[0] for b in bounds]
    his = [b[1] for b in bounds]
    return (
        tuple(min(b[k] for b in los) for k in range(3)),
        tuple(max(b[k] for b in his) for k in range(3)),
    )


def _mid_area(b0: tuple, b1: tuple) -> float:
    """t=0.5 时刻包围盒表面积（SAH 代价用）；退化边按 0 计。"""
    edges = []
    for k in range(3):
        lo = 0.5 * b0[0][k] + 0.5 * b1[0][k]
        hi = 0.5 * b0[1][k] + 0.5 * b1[1][k]
        edge = hi - lo
        edges.append(edge if edge > 0.0 else 0.0)
    ex, ey, ez = edges
    return 2.0 * (ex * ey + ey * ez + ez * ex)


def _stable_mid_split(
    ids: list[int], centroid_mid: Sequence[tuple[float, float, float]], axis: int
) -> tuple[list[int], list[int]]:
    """退化质心（或桶坍缩）时的稳定中分：按 (质心坐标, face_id) 排序后切半。

    加 face_id 作次序键，保证输入完全重合时划分仍然确定。
    """
    order = sorted(ids, key=lambda face_id: (centroid_mid[face_id][axis], face_id))
    cut = len(order) // 2
    return order[:cut], order[cut:]


def _grow(bounds: tuple, tb: tuple) -> tuple:
    return (
        tuple(min(bounds[0][k], tb[0][k]) for k in range(3)),
        tuple(max(bounds[1][k], tb[1][k]) for k in range(3)),
    )


def _empty_bounds() -> tuple:
    return ((float("inf"),) * 3, (float("-inf"),) * 3)


def _build_recursive(
    ids: list[int],
    tri_b0: Sequence[tuple],
    tri_b1: Sequence[tuple],
    centroid_mid: Sequence[tuple[float, float, float]],
) -> tuple[_Node, int]:
    b0 = _union_bounds([tri_b0[i] for i in ids])
    b1 = _union_bounds([tri_b1[i] for i in ids])

    if len(ids) <= LEAF_MAX:
        return _Node(b0, b1, None, None, tuple(sorted(ids)), -1), 1

    # 用 t=0.5 的质心界选轴。
    centroid_lo = tuple(min(centroid_mid[i][k] for i in ids) for k in range(3))
    centroid_hi = tuple(max(centroid_mid[i][k] for i in ids) for k in range(3))
    extents = tuple(centroid_hi[k] - centroid_lo[k] for k in range(3))
    axis = max(range(3), key=lambda k: extents[k])

    split_left: Optional[list[int]] = None
    split_right: Optional[list[int]] = None

    if extents[axis] > 0.0:
        parent_area = _mid_area(b0, b1)
        if parent_area > 0.0:
            span = centroid_hi[axis] - centroid_lo[axis]
            inv_span = 1.0 / span
            counts = [0] * BIN_COUNT
            bin_lo = [[[float("inf")] * 3, [float("inf")] * 3] for _ in range(BIN_COUNT)]
            bin_hi = [[[float("-inf")] * 3, [float("-inf")] * 3] for _ in range(BIN_COUNT)]
            assignment: dict[int, int] = {}

            for face_id in ids:
                scaled = (centroid_mid[face_id][axis] - centroid_lo[axis]) * inv_span
                bucket = int(scaled * BIN_COUNT)
                if bucket < 0:
                    bucket = 0
                elif bucket >= BIN_COUNT:
                    bucket = BIN_COUNT - 1
                assignment[face_id] = bucket
                counts[bucket] += 1
                for endpoint, tb in ((0, tri_b0[face_id]), (1, tri_b1[face_id])):
                    for k in range(3):
                        if tb[0][k] < bin_lo[bucket][endpoint][k]:
                            bin_lo[bucket][endpoint][k] = tb[0][k]
                        if tb[1][k] > bin_hi[bucket][endpoint][k]:
                            bin_hi[bucket][endpoint][k] = tb[1][k]

            # 前缀界 [0..split) 与后缀界 [split..8)，均只由落入桶内的三角形求出。
            pref_count = [0] * (BIN_COUNT + 1)
            pref_b0: list[tuple] = [_empty_bounds()] * (BIN_COUNT + 1)
            pref_b1: list[tuple] = [_empty_bounds()] * (BIN_COUNT + 1)
            for bucket in range(BIN_COUNT):
                pref_count[bucket + 1] = pref_count[bucket] + counts[bucket]
                if counts[bucket] == 0:
                    pref_b0[bucket + 1] = pref_b0[bucket]
                    pref_b1[bucket + 1] = pref_b1[bucket]
                else:
                    pref_b0[bucket + 1] = _grow(
                        pref_b0[bucket], (tuple(bin_lo[bucket][0]), tuple(bin_hi[bucket][0]))
                    )
                    pref_b1[bucket + 1] = _grow(
                        pref_b1[bucket], (tuple(bin_lo[bucket][1]), tuple(bin_hi[bucket][1]))
                    )

            suff_count = [0] * (BIN_COUNT + 1)
            suff_b0: list[tuple] = [_empty_bounds()] * (BIN_COUNT + 1)
            suff_b1: list[tuple] = [_empty_bounds()] * (BIN_COUNT + 1)
            for bucket in range(BIN_COUNT - 1, -1, -1):
                suff_count[bucket] = suff_count[bucket + 1] + counts[bucket]
                if counts[bucket] == 0:
                    suff_b0[bucket] = suff_b0[bucket + 1]
                    suff_b1[bucket] = suff_b1[bucket + 1]
                else:
                    suff_b0[bucket] = _grow(
                        suff_b0[bucket + 1], (tuple(bin_lo[bucket][0]), tuple(bin_hi[bucket][0]))
                    )
                    suff_b1[bucket] = _grow(
                        suff_b1[bucket + 1], (tuple(bin_lo[bucket][1]), tuple(bin_hi[bucket][1]))
                    )

            total = len(ids)
            best_cost = math.inf
            best_split = -1
            for split in range(1, BIN_COUNT):
                left_count = pref_count[split]
                right_count = suff_count[split]
                if left_count == 0 or right_count == 0:
                    continue
                # 简化 SAH：代价 = nL*AL + nR*AR（已除以父表面积），叶代价 = total。
                cost = (
                    left_count * _mid_area(pref_b0[split], pref_b1[split])
                    + right_count * _mid_area(suff_b0[split], suff_b1[split])
                ) / parent_area
                if cost < best_cost:
                    best_cost = cost
                    best_split = split

            # 不划算或桶坍缩（best_split 仍 -1）则落到稳定中分。
            if best_split > 0 and best_cost < total:
                left_ids = [i for i in ids if assignment[i] < best_split]
                right_ids = [i for i in ids if assignment[i] >= best_split]
                if left_ids and right_ids:
                    split_left, split_right = left_ids, right_ids

    if split_left is None:
        split_left, split_right = _stable_mid_split(ids, centroid_mid, axis)

    left_node, left_count = _build_recursive(split_left, tri_b0, tri_b1, centroid_mid)
    right_node, right_count = _build_recursive(split_right, tri_b0, tri_b1, centroid_mid)
    return _Node(b0, b1, left_node, right_node, None, axis), left_count + right_count + 1


# ---------------------------------------------------------------------------
# MotionBVH 句柄
# ---------------------------------------------------------------------------


class MotionBVH:
    """build() 的结果；不可再变，可被多线程同时调用 intersect（自身无遍历状态）。"""

    __slots__ = ("_triangles", "_root", "triangle_count", "node_count")

    def __init__(
        self,
        triangles: tuple,
        root: Optional[_Node],
        triangle_count: int,
        node_count: int,
    ) -> None:
        self._triangles = triangles
        self._root = root
        self.triangle_count = triangle_count
        self.node_count = node_count

    def intersect(self, ray: Ray, time: float, *, budget: Optional[int] = None) -> Optional[HitResult]:
        return intersect(self, ray, time, budget=budget)

    def intersect_full(
        self, ray: Ray, time: float, *, budget: Optional[int] = None
    ) -> tuple[Optional[HitResult], TraceStats]:
        return intersect_full(self, ray, time, budget=budget)


def build(triangles: Sequence) -> MotionBVH:
    """由三角形序列构建运动 BVH。

    纯函数：成功返回不可再变的 :class:`MotionBVH`；任何输入非法都抛
    :class:`MotionBVHError`，不会产生部分构建结果。
    """
    if not isinstance(triangles, (tuple, list)):
        raise MotionBVHError(
            f"triangles: expected a list/tuple, got {type(triangles).__name__}"
        )
    if len(triangles) > MAX_TRIANGLES:
        raise MotionBVHError(
            f"triangles: length {len(triangles)} exceeds the interface limit {MAX_TRIANGLES}"
        )

    # 先完整校验并规范化输入，再开始任何构建状态。
    canonical = tuple(_coerce_triangle(tri, i) for i, tri in enumerate(triangles))
    if not canonical:
        return MotionBVH(canonical, None, 0, 0)

    tri_b0 = [_tri_bounds(pair[0]) for pair in canonical]
    tri_b1 = [_tri_bounds(pair[1]) for pair in canonical]
    centroid_mid = []
    for v0, v1 in canonical:
        c0 = tuple((v0[0][k] + v0[1][k] + v0[2][k]) / 3.0 for k in range(3))
        c1 = tuple((v1[0][k] + v1[1][k] + v1[2][k]) / 3.0 for k in range(3))
        centroid_mid.append(tuple(0.5 * c0[k] + 0.5 * c1[k] for k in range(3)))

    root, node_count = _build_recursive(
        list(range(len(canonical))), tri_b0, tri_b1, centroid_mid
    )
    return MotionBVH(canonical, root, len(canonical), node_count)


# ---------------------------------------------------------------------------
# 遍历
# ---------------------------------------------------------------------------


def _box_hit(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    b0: tuple,
    b1: tuple,
    time: float,
    tmin: float,
    tmax: float,
) -> bool:
    """Slab 测试；包围盒先按时间插值再做 GUARD 级外扩。支持零方向分量。"""
    nearest = tmin
    farthest = tmax
    s = 1.0 - time
    for k in range(3):
        lo = b0[0][k] * s + b1[0][k] * time
        hi = b0[1][k] * s + b1[1][k] * time
        margin = GUARD * max(1.0, abs(lo), abs(hi))
        lo -= margin
        hi += margin
        d = direction[k]
        o = origin[k]
        if d == 0.0:
            if o < lo or o > hi:
                return False
            continue
        t_lo = (lo - o) / d
        t_hi = (hi - o) / d
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        if t_lo > nearest:
            nearest = t_lo
        if t_hi < farthest:
            farthest = t_hi
        if nearest > farthest:
            return False
    return True


def _triangle_hit(
    tri: tuple,
    time: float,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    tmin: float,
    tmax: float,
):
    """Möller–Trumbore（双面），返回 (t, u, v, w) 或 None。

    当前时刻零面积（det==0）或与射线平行的三角形按未命中处理。
    """
    (a0, b0, c0), (a1, b1, c1) = tri
    s = 1.0 - time
    a = tuple(a0[k] * s + a1[k] * time for k in range(3))
    b = tuple(b0[k] * s + b1[k] * time for k in range(3))
    c = tuple(c0[k] * s + c1[k] * time for k in range(3))

    edge1 = _sub(b, a)
    edge2 = _sub(c, a)
    h = _cross(direction, edge2)
    det = _dot(edge1, h)
    if det == 0.0:
        return None
    inv_det = 1.0 / det
    svec = _sub(origin, a)
    bweight = _dot(svec, h) * inv_det  # MT 的 u：沿 edge1（顶点 b）的权重
    q = _cross(svec, edge1)
    cweight = _dot(direction, q) * inv_det  # MT 的 v：沿 edge2（顶点 c）的权重

    if bweight < -GUARD or cweight < -GUARD or bweight + cweight > 1.0 + GUARD:
        return None
    t = _dot(edge2, q) * inv_det
    t_guard = GUARD * max(1.0, abs(t))
    if t < tmin - t_guard or t > tmax + t_guard:
        return None

    # GUARD 内的边值越界夹紧并归一化，输出干净重心坐标，顺序为 (a, b, c)。
    bweight = max(bweight, 0.0)
    cweight = max(cweight, 0.0)
    total_bc = bweight + cweight
    if total_bc > 1.0:
        bweight /= total_bc
        cweight /= total_bc
        total_bc = 1.0
    aweight = 1.0 - total_bc
    if aweight < 0.0:
        aweight = 0.0
    return t, aweight, bweight, cweight


def intersect(
    bvh: MotionBVH, ray: Ray, time: float, *, budget: Optional[int] = None
) -> Optional[HitResult]:
    """在给定时刻对 BVH 做最近命中遍历。

    * 命中返回 :class:`HitResult`；
    * 完整遍历后证明区间内无命中，返回 ``None``；
    * 给定 ``budget``（正整数，按弹出节点数计）且预算耗尽时抛
      :class:`BudgetExhaustedError` —— 与“证明无命中”严格区分。

    miss 时若仍需访问计数，改用 :func:`intersect_full`。
    """
    hit, _stats = intersect_full(bvh, ray, time, budget=budget)
    return hit


def intersect_full(
    bvh: MotionBVH, ray: Ray, time: float, *, budget: Optional[int] = None
) -> tuple[Optional[HitResult], TraceStats]:
    """同 :func:`intersect`，但始终同时返回 ``(命中或 None, TraceStats)``。

    预算耗尽时仍抛 :class:`BudgetExhaustedError`（异常对象携带耗尽前计数）。
    """
    if not isinstance(bvh, MotionBVH):
        raise MotionBVHError(
            f"bvh: expected a MotionBVH from build(), got {type(bvh).__name__}"
        )
    if not isinstance(ray, Ray):
        raise MotionBVHError(f"ray: expected a Ray, got {type(ray).__name__}")
    query_time = _query_time(time)
    step_budget = _budget(budget)

    root = bvh._root
    if root is None:
        return None, TraceStats(0, 0)

    origin, direction = ray.origin, ray.direction
    tmin = ray.tmin
    best_t = ray.tmax  # 用当前最近距离持续收窄裁剪区间。
    best_id = -1
    best_bary: Optional[tuple[float, float, float]] = None
    nodes_visited = 0
    triangle_tests = 0

    stack = [root]
    while stack:
        node = stack.pop()
        nodes_visited += 1
        if step_budget is not None and nodes_visited > step_budget:
            raise BudgetExhaustedError(
                f"traversal budget exhausted after {nodes_visited - 1} node visit(s); "
                "no proven miss/hit result is available",
                nodes_visited - 1,
                triangle_tests,
            )

        if not _box_hit(origin, direction, node.b0, node.b1, query_time, tmin, best_t):
            continue

        if node.ids is not None:
            for face_id in node.ids:  # 叶内按 face_id 升序
                triangle_tests += 1
                hit = _triangle_hit(
                    bvh._triangles[face_id], query_time, origin, direction, tmin, best_t
                )
                if hit is None:
                    continue
                t, u, v, w = hit
                if best_id < 0:
                    best_t, best_id, best_bary = t, face_id, (u, v, w)
                    continue
                band = TIE_EPS * max(1.0, abs(t), abs(best_t))
                if t < best_t - band:
                    # 严格更近：另起同距簇。
                    best_t, best_id, best_bary = t, face_id, (u, v, w)
                elif t <= best_t + band and face_id < best_id:
                    # 同距簇内：face_id 取小，簇距离取小，与遍历先后顺序无关。
                    best_id = face_id
                    best_bary = (u, v, w)
                    if t < best_t:
                        best_t = t
        else:
            # 近端先弹出：沿分裂轴按方向符号决定远近（零分量时左侧为近）。
            near, far = node.left, node.right
            if direction[node.axis] < 0.0:
                near, far = far, near
            stack.append(far)
            stack.append(near)

    stats = TraceStats(triangle_tests, nodes_visited)
    if best_id < 0:
        return None, stats
    # 边界命中经 GUARD 接受后，t 夹紧回声明的查询区间。
    clamped_t = min(max(best_t, tmin), ray.tmax)
    return HitResult(best_id, clamped_t, best_bary, triangle_tests, nodes_visited), stats
