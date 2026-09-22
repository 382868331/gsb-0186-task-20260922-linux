"""motion_bvh — 带运动界的 BVH 射线遍历可复用模块（纯标准库）。

契约概览（详见 README.md）：
  * build(triangles) -> BVH；每三角形给出 t=0 与 t=1 两端顶点，顶点在 t∈[0,1]
    线性运动。最多 300 个三角形。
  * BVH 节点保存 t=0/t=1 两端 AABB；遍历时刻 τ 对包围盒线性插值（保守外包）。
  * 构建使用参考时刻 t=0 质心、固定 8 桶 SAH；质心退化时使用稳定中分；
    叶节点至多 4 个三角形。
  * intersect 支持 t_min/t_max、零方向分量（零向量方向非法）、同距按 face_id
    较小者胜；输出重心坐标与访问计数。
  * 所有数值输入拒绝 NaN/Infinity；bool 不作为整数/数值接受。

本模块只依赖 Python 3.14 标准库。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "Triangle",
    "Ray",
    "Intersection",
    "BVH",
    "build",
    "intersect",
]

# ---------------------------------------------------------------------------
# 公开常量
# ---------------------------------------------------------------------------

MAX_TRIANGLES = 300
"""build 接受的三角形数上限。"""

LEAF_SIZE = 4
"""叶节点三角形数上限。"""

BIN_COUNT = 8
"""SAH 固定桶数。"""

_TIME_LO, _TIME_HI = 0.0, 1.0

# 机器精度量级常量（误差口径见 README）
_EPS = 2.220446049250313e-16
# 重心坐标边界接受宽限（绝对值），仅用于吸收个位 ulp 的舍入
_BARY_EPS = 1e-12
# 同距判定：16 ulp 的相对容差（t 尺度自适应），同距内按 face_id 决胜
_TIE_ULPS = 16
# 插值包围盒的亚 ulp 防误剔除护栏（相对尺度），仅用于抵消 lerp 舍入
_BOX_GUARD_ULPS = 4


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _is_real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_real(x, where: str) -> float:
    """校验单个标量：必须是 int/float（排除 bool）、有限；返回 float。"""
    if not _is_real(x):
        raise TypeError(f"{where} 必须是实数(int/float)，得到 {type(x).__name__}")
    f = float(x)
    if math.isnan(f):
        raise ValueError(f"{where} 不能是 NaN")
    if math.isinf(f):
        raise ValueError(f"{where} 不能是 Infinity")
    return f


def _check_point(p, where: str):
    if not isinstance(p, (tuple, list)):
        raise TypeError(f"{where} 必须是长度为 3 的序列(tuple/list)")
    if len(p) != 3:
        raise TypeError(f"{where} 必须恰好含 3 个分量，得到 {len(p)}")
    return (
        _check_real(p[0], f"{where}[0]"),
        _check_real(p[1], f"{where}[1]"),
        _check_real(p[2], f"{where}[2]"),
    )


def _check_int(x, where: str) -> int:
    if not isinstance(x, int) or isinstance(x, bool):
        raise TypeError(f"{where} 必须是 int（拒绝 bool），得到 {type(x).__name__}")
    return x


# ---------------------------------------------------------------------------
# 向量小工具（内部，tuple 表示）
# ---------------------------------------------------------------------------

def _vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vcross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vlerp(a, b, t):
    """t=0 取 a，t=1 取 b。"""
    return (
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    )


def _union_box(box, p):
    lo, hi = box
    return (
        (min(lo[0], p[0]), min(lo[1], p[1]), min(lo[2], p[2])),
        (max(hi[0], p[0]), max(hi[1], p[1]), max(hi[2], p[2])),
    )


_INF = math.inf
_EMPTY_BOX = ((_INF, _INF, _INF), (-_INF, -_INF, -_INF))


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Triangle:
    """一个运动三角形。

    a0,b0,c0 为 t=0 三顶点；a1,b1,c1 为 t=1 三顶点。顶点位置随时间
    p(τ) = p0 + τ*(p1-p0) 线性运动。face_id 为面标识（同距决胜用），
    缺省由 build 按输入顺序赋 0..N-1；显式给出时必须为非负且互不相同。
    需要静止三角形可用 Triangle.stationary。
    """

    a0: tuple
    b0: tuple
    c0: tuple
    a1: tuple
    b1: tuple
    c1: tuple
    face_id: Optional[int] = None

    @staticmethod
    def stationary(a, b, c, face_id: Optional[int] = None) -> "Triangle":
        return Triangle(a, b, c, a, b, c, face_id)


@dataclass(frozen=True, slots=True)
class Ray:
    """射线 origin + t*direction。direction 允许含零分量，但不能为零向量；
    不必归一化，命中参数 t 以 direction 为单位。"""

    origin: tuple
    direction: tuple


@dataclass(frozen=True, slots=True)
class Intersection:
    """相交查询结果。未命中时 hit=False，计数字段仍有效（证明无解的遍历量）。

    hit              是否命中
    face_id          命中三角形标识；未命中为 None
    t                射线参数（origin + t*direction）
    u, v             重心坐标：P = (1-u-v)*a + u*b + v*c
    point            命中点
    nodes_visited    执行了包围盒测试的节点数（含根）
    triangles_tested 实际做三角形相交测试的面数
    """

    hit: bool
    face_id: Optional[int]
    t: float
    u: float
    v: float
    point: Optional[tuple]
    nodes_visited: int
    triangles_tested: int

    def __bool__(self) -> bool:
        return self.hit


# ---------------------------------------------------------------------------
# BVH 内部结构
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("lo0", "hi0", "lo1", "hi1", "g_eps", "left", "right", "faces")

    def __init__(self):
        # 两端 AABB
        self.lo0 = self.hi0 = self.lo1 = self.hi1 = None
        # 与重心边界宽限配套的空间外扩量（构建后固定）
        self.g_eps = 0.0
        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None
        self.faces: Optional[tuple] = None  # 叶节点：(BVH面序号,...)


class BVH:
    """不可变的运动层次包围盒。请通过 motion_bvh.build 构造。"""

    __slots__ = ("_faces", "_face_ids", "_root", "_count")

    def __init__(self):
        raise TypeError("BVH 不可直接构造，请使用 motion_bvh.build()")

    @classmethod
    def _from_root(cls, faces, face_ids, root, count):
        self = object.__new__(cls)
        self._faces = faces          # [(a0,b0,c0,a1,b1,c1), ...]
        self._face_ids = face_ids    # 输入序号 -> face_id
        self._root = root
        self._count = count
        return self

    @property
    def triangle_count(self) -> int:
        return self._count

    # ---- 公开查询入口 -----------------------------------------------------

    def intersect(
        self,
        ray: Ray,
        time: float,
        t_min: float = 0.0,
        t_max: float = math.inf,
    ) -> Intersection:
        return _intersect(self, ray, time, t_min, t_max)


# 模块级函数也直接暴露 build/intersect 两个名字

def build(triangles: Sequence[Triangle]) -> BVH:
    """由运动三角形列表构建 BVH。

    全量校验通过后才构建：非法输入时不会产生任何对象（全有或全无）。
    空列表合法，得到空 BVH（任意查询均未命中）。
    """
    if not isinstance(triangles, (list, tuple)):
        # 允许可迭代对象，但先物化以便计数与定位
        try:
            triangles = list(triangles)
        except TypeError as exc:
            raise TypeError("triangles 必须是 Triangle 的可迭代对象") from exc
    n = len(triangles)
    if n > MAX_TRIANGLES:
        raise ValueError(
            f"三角形数 {n} 超过上限 {MAX_TRIANGLES}（接口边界，非性能建议）"
        )

    faces = []
    seen_ids = set()
    for i, tri in enumerate(triangles):
        where = f"triangles[{i}]"
        if not isinstance(tri, Triangle):
            raise TypeError(
                f"{where} 必须是 motion_bvh.Triangle，得到 {type(tri).__name__}"
            )
        a0 = _check_point(tri.a0, f"{where}.a0")
        b0 = _check_point(tri.b0, f"{where}.b0")
        c0 = _check_point(tri.c0, f"{where}.c0")
        a1 = _check_point(tri.a1, f"{where}.a1")
        b1 = _check_point(tri.b1, f"{where}.b1")
        c1 = _check_point(tri.c1, f"{where}.c1")
        if tri.face_id is None:
            fid = i
        else:
            fid = _check_int(tri.face_id, f"{where}.face_id")
            if fid < 0:
                raise ValueError(f"{where}.face_id 必须非负，得到 {fid}")
        if fid in seen_ids:
            raise ValueError(f"{where}.face_id={fid} 与前面的三角形重复")
        seen_ids.add(fid)
        faces.append((a0, b0, c0, a1, b1, c1, fid))

    # 校验全部通过，开始构建（不会再失败）
    geoms = [entry[:6] for entry in faces]
    face_ids = [entry[6] for entry in faces]
    root = _build(geoms, list(range(n))) if n else None
    return BVH._from_root(geoms, face_ids, root, n)


def intersect(
    bvh: BVH,
    ray: Ray,
    time: float,
    t_min: float = 0.0,
    t_max: float = math.inf,
) -> Intersection:
    """模块级入口，等价于 bvh.intersect(ray, time, t_min, t_max)。"""
    return _intersect(bvh, ray, time, t_min, t_max)


# ---------------------------------------------------------------------------
# 构建：8 桶 SAH + 稳定中分兜底
# ---------------------------------------------------------------------------

def _face_endpoint_boxes(face):
    a0, b0, c0, a1, b1, c1 = face
    box0 = _union_box(_union_box(_EMPTY_BOX, a0), b0)
    box0 = _union_box(box0, c0)
    box1 = _union_box(_union_box(_EMPTY_BOX, a1), b1)
    box1 = _union_box(box1, c1)
    return box0, box1


def _centroid0(face):
    a0, b0, c0 = face[0], face[1], face[2]
    return (
        (a0[0] + b0[0] + c0[0]) / 3.0,
        (a0[1] + b0[1] + c0[1]) / 3.0,
        (a0[2] + b0[2] + c0[2]) / 3.0,
    )


def _box_surface_area(lo, hi) -> float:
    d = _vsub(hi, lo)
    # 负分量（空盒）防御：正常构建路径不会出现
    if d[0] <= 0.0 or d[1] <= 0.0 or d[2] <= 0.0:
        return 0.0
    return 2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])


def _merge_node_boxes(node, box0, box1):
    """把两端盒并入节点。"""
    lo0, hi0 = box0
    lo1, hi1 = box1
    if node.lo0 is None:
        node.lo0, node.hi0, node.lo1, node.hi1 = lo0, hi0, lo1, hi1
        return
    node.lo0 = (min(node.lo0[0], lo0[0]), min(node.lo0[1], lo0[1]),
                min(node.lo0[2], lo0[2]))
    node.hi0 = (max(node.hi0[0], hi0[0]), max(node.hi0[1], hi0[1]),
                max(node.hi0[2], hi0[2]))
    node.lo1 = (min(node.lo1[0], lo1[0]), min(node.lo1[1], lo1[1]),
                min(node.lo1[2], lo1[2]))
    node.hi1 = (max(node.hi1[0], hi1[0]), max(node.hi1[1], hi1[1]),
                max(node.hi1[2], hi1[2]))


def _build(geoms, indices):
    node = _Node()
    boxes = []
    cents = []
    for idx in indices:
        b0, b1 = _face_endpoint_boxes(geoms[idx])
        boxes.append((b0, b1))
        cents.append(_centroid0(geoms[idx]))
        _merge_node_boxes(node, b0, b1)

    n = len(indices)
    if n <= LEAF_SIZE:
        node.faces = tuple(indices)
        node.g_eps = _node_guard(node)
        return node
    # 质心包围盒（参考时刻 t=0）
    clo = [math.inf] * 3
    chi = [-math.inf] * 3
    for c in cents:
        for k in range(3):
            if c[k] < clo[k]:
                clo[k] = c[k]
            if c[k] > chi[k]:
                chi[k] = c[k]
    extent = [chi[k] - clo[k] for k in range(3)]
    axis = 0
    if extent[1] > extent[axis]:
        axis = 1
    if extent[2] > extent[axis]:
        axis = 2

    split = _sah_split(indices, boxes, cents, axis, clo[axis], chi[axis])
    if split is None:
        split = _mid_split(indices, cents, axis)

    left_idx, right_idx = split
    node.left = _build(geoms, left_idx)
    node.right = _build(geoms, right_idx)
    # 内部节点盒包含两子盒，其两端最大对角不小于任一子节点，护栏用更大尺度
    node.g_eps = _node_guard(node)
    return node


def _node_guard(node):
    """与三角形重心边界宽限 _BARY_EPS 配套的盒外扩量。

    三角形谓词会接受边界外最多约 _BARY_EPS 倍三角形尺度的点；为保证
    “盒谓词命中”是“任一三角形谓词命中”的必要条件（不会把该测的叶剔除），
    每个节点的插值盒在各轴向外扩 g_eps。取两端盒对角线的最大分量，故对
    任意 τ 的插值盒都保守。
    """
    diag0 = max(node.hi0[k] - node.lo0[k] for k in range(3))
    diag1 = max(node.hi1[k] - node.lo1[k] for k in range(3))
    scale = max(1.0, diag0, diag1)
    # 系数 4：覆盖 Δu·|e1|+Δv·|e2| 的最坏组合（边长 ≤ √3·diag，Δ 之和 ≤ 2eps）
    return 4.0 * _BARY_EPS * scale


def _sah_split(indices, boxes, cents, axis, cmin, cmax):
    """固定 8 桶 SAH，返回 (left, right) 或 None（退化/无法划分）。"""
    n = len(indices)
    if not (cmax > cmin):
        return None

    inv = 1.0 / (cmax - cmin)
    bin_count = [0] * BIN_COUNT
    bin_lo0 = [[math.inf] * 3 for _ in range(BIN_COUNT)]
    bin_hi0 = [[-math.inf] * 3 for _ in range(BIN_COUNT)]

    for j, idx in enumerate(indices):
        x = cents[j][axis]
        k = int(BIN_COUNT * (x - cmin) * inv)
        if k >= BIN_COUNT:
            k = BIN_COUNT - 1
        bin_count[k] += 1
        (lo0, hi0), _b1 = boxes[j]
        dst_lo = bin_lo0[k]
        dst_hi = bin_hi0[k]
        for q in range(3):
            if lo0[q] < dst_lo[q]:
                dst_lo[q] = lo0[q]
            if hi0[q] > dst_hi[q]:
                dst_hi[q] = hi0[q]

    # 前缀/后缀扫描
    prefix = [None] * (BIN_COUNT - 1)
    cnt = 0
    plo = [math.inf] * 3
    phi = [-math.inf] * 3
    for k in range(BIN_COUNT - 1):
        cnt += bin_count[k]
        for q in range(3):
            if bin_lo0[k][q] < plo[q]:
                plo[q] = bin_lo0[k][q]
            if bin_hi0[k][q] > phi[q]:
                phi[q] = bin_hi0[k][q]
        if cnt:
            prefix[k] = (cnt, _box_surface_area(plo, phi))

    suffix = [None] * (BIN_COUNT - 1)
    cnt = 0
    slo = [math.inf] * 3
    shi = [-math.inf] * 3
    for k in range(BIN_COUNT - 1, 0, -1):
        cnt += bin_count[k]
        for q in range(3):
            if bin_lo0[k][q] < slo[q]:
                slo[q] = bin_lo0[k][q]
            if bin_hi0[k][q] > shi[q]:
                shi[q] = bin_hi0[k][q]
        if cnt:
            suffix[k - 1] = (cnt, _box_surface_area(slo, shi))

    # 父 SA 取整体两端合并（t=0）；这里直接用所有 bin 盒
    par_lo = [math.inf] * 3
    par_hi = [-math.inf] * 3
    for k in range(BIN_COUNT):
        for q in range(3):
            if bin_lo0[k][q] < par_lo[q]:
                par_lo[q] = bin_lo0[k][q]
            if bin_hi0[k][q] > par_hi[q]:
                par_hi[q] = bin_hi0[k][q]
    parent_sa = _box_surface_area(par_lo, par_hi)
    if parent_sa <= 0.0:
        return None  # 全为共面/点的退化几何，SAH 无意义

    best_cost = math.inf
    best_plane = -1
    for k in range(BIN_COUNT - 1):
        lp = prefix[k]
        rp = suffix[k]
        if lp is None or rp is None:
            continue
        cost = 1.0 + (lp[0] * lp[1] + rp[0] * rp[1]) / parent_sa
        if cost < best_cost:
            best_cost = cost
            best_plane = k

    if best_plane < 0:
        return None  # 全部质心落同一桶，无法用桶平面划分

    # 按 best_plane 对应桶边界做实际划分
    cut = cmin + (best_plane + 1) * (cmax - cmin) / BIN_COUNT
    left, right = [], []
    for j, idx in enumerate(indices):
        if cents[j][axis] < cut:
            left.append(idx)
        else:
            right.append(idx)
    if not left or not right:
        return None

    # n <= LEAF_SIZE 时不会进入本函数；n > LEAF_SIZE：
    # 若 SAH 成本不优于叶成本（n），按题意退化策略仍须稳定中分而非违反叶上限，
    # 故仅当 SAH 确实更省时采用，否则交给中分兜底。
    leaf_cost = float(n)
    if best_cost >= leaf_cost:
        return None
    return left, right


def _mid_split(indices, cents, axis):
    """稳定中分：按轴质心排序（平手按输入序号），取 n//2。

    质心在所有轴都退化时 axis 为 0，但排序键加入序号后仍能确定地均分。
    """
    order = sorted(
        range(len(indices)),
        key=lambda j: (cents[j][axis], cents[j][(axis + 1) % 3],
                       cents[j][(axis + 2) % 3], indices[j]),
    )
    half = len(order) // 2
    left = [indices[order[j]] for j in range(half)]
    right = [indices[order[j]] for j in range(half, len(order))]
    return left, right


# ---------------------------------------------------------------------------
# 遍历
# ---------------------------------------------------------------------------

def _intersect(bvh: BVH, ray: Ray, time, t_min, t_max) -> Intersection:
    if not isinstance(bvh, BVH):
        raise TypeError("ray 查询目标必须是 motion_bvh.BVH")
    if not isinstance(ray, Ray):
        raise TypeError("ray 必须是 motion_bvh.Ray")
    tau = _check_real(time, "time")
    if not (_TIME_LO <= tau <= _TIME_HI):
        raise ValueError(f"time 必须在 [{_TIME_LO}, {_TIME_HI}] 内，得到 {tau}")
    t0 = _check_real(t_min, "t_min")
    # t_max 特殊：允许 +Infinity（默认无界），拒绝 bool / NaN / -Infinity
    if not _is_real(t_max):
        raise TypeError(
            f"t_max 必须是实数(int/float)，得到 {type(t_max).__name__}"
        )
    t1 = float(t_max)
    if math.isnan(t1):
        raise ValueError("t_max 不能是 NaN")
    if math.isinf(t1) and t1 < 0.0:
        raise ValueError("t_max 不能是 -Infinity")
    if t0 > t1:
        raise ValueError(f"t_min({t0}) 不能大于 t_max({t1})")
    origin = _check_point(ray.origin, "ray.origin")
    if not isinstance(ray.direction, (tuple, list)) or len(ray.direction) != 3:
        raise TypeError("ray.direction 必须是长度为 3 的序列")
    direction = _check_point(ray.direction, "ray.direction")
    if direction == (0.0, 0.0, 0.0):
        raise ValueError("ray.direction 不能为零向量")

    root = bvh._root
    if root is None or root.lo0 is None:
        return Intersection(False, None, math.nan, math.nan, math.nan, None, 0, 0)

    nodes_visited = 0
    triangles_tested = 0

    # abs_best：已见严格最小 t（顺序无关）；win：同距带内的决胜候选
    # (t,u,v,face_id,face_idx)。两阶段规则保证与遍历顺序无关：
    # 最终赢家 = 与全局最小 t 相差 ≤ tie_tol 的候选中 face_id 最小者。
    abs_best = t1
    win = None

    def tie_tol(a, b):
        return _TIE_ULPS * _EPS * max(1.0, abs(a), abs(b))

    stack = [root]
    while stack:
        node = stack.pop()
        nodes_visited += 1
        # 剪枝窗口向外放一个同距容差带，确保同距面不会因当前最近距离
        # 被提前剔除（“证明无同距面”与预算无关，这里不存在预算耗尽）。
        # 同距带定义为 16 ulp；窗口再加 2 ulp 余量，覆盖“tc 在 t* 带内”
        # 反解出的 tc 上界（t*/(1−16eps) 比 t*+16eps 高 O(eps²)）。
        if math.isinf(abs_best):
            window = abs_best
        else:
            window = abs_best + (_TIE_ULPS + 2) * _EPS * max(
                1.0, abs(abs_best))
        if not _box_hit(node, origin, direction, tau, t0, window):
            continue
        if node.faces is not None:
            for idx in node.faces:
                triangles_tested += 1
                face = bvh._faces[idx]
                hit = _triangle_hit(face, origin, direction, tau, t0, window)
                if hit is None:
                    continue
                ht, hu, hv = hit
                # 同距带仅放宽剪枝/测试窗口以避免漏同距面；用户窗口
                # [t0,t1] 是硬契约边界，越窗命中不得成为候选。
                if ht > t1:
                    continue
                fid = bvh._face_ids[idx]
                if win is None:
                    abs_best = ht
                    win = (ht, hu, hv, fid, idx)
                elif ht < abs_best:
                    abs_best = ht
                    band = tie_tol(win[0], ht)
                    if win[0] > ht + band or fid < win[3]:
                        win = (ht, hu, hv, fid, idx)
                else:
                    band = tie_tol(ht, abs_best)
                    if abs(ht - abs_best) <= band and fid < win[3]:
                        win = (ht, hu, hv, fid, idx)
        else:
            # 左子先测（仅影响剪枝效率，不影响结果）
            stack.append(node.right)
            stack.append(node.left)

    if win is None:
        return Intersection(
            False, None, math.nan, math.nan, math.nan, None,
            nodes_visited, triangles_tested,
        )

    best_t, best_u, best_v, best_id, best_idx = win
    face = bvh._faces[best_idx]
    a = _vlerp(face[0], face[3], tau)
    b = _vlerp(face[1], face[4], tau)
    c = _vlerp(face[2], face[5], tau)
    u, v = best_u, best_v
    w = 1.0 - u - v
    # 命中点按重心组合（与 origin+t*d 同源数据，二者在容差内一致）
    px = w * a[0] + u * b[0] + v * c[0]
    py = w * a[1] + u * b[1] + v * c[1]
    pz = w * a[2] + u * b[2] + v * c[2]
    return Intersection(
        True, best_id, best_t, u, v, (px, py, pz),
        nodes_visited, triangles_tested,
    )


def _box_hit(node, o, d, tau, t_min, t_max) -> bool:
    """Slab 法射线-AABB（盒为两端 AABB 在 τ 的线性插值）。

    * 方向零分量单独处理（原点在板外即不命中）。
    * 相交区间端点包含边界（贴边/贴面算入）。
    * 插值盒向外放 _BOX_GUARD_ULPS ulp，仅抵消 lerp 与逐顶点运动之间的
      舍入，不改变几何包含关系（端点盒是顶点端点位置的精确 min/max）。
    """
    lo0, hi0, lo1, hi1 = node.lo0, node.hi0, node.lo1, node.hi1
    tnear, tfar = t_min, t_max
    for k in range(3):
        lo = lo0[k] + tau * (lo1[k] - lo0[k])
        hi = hi0[k] + tau * (hi1[k] - hi0[k])
        # 外扩 = 重心边界宽限配套量（随盒尺度）+ 个位 ulp 的 lerp 舍入护栏，
        # 仅用于保证谓词包含关系，不用它掩盖几何边界外的错误（详见 README）。
        guard = node.g_eps + _BOX_GUARD_ULPS * _EPS * max(1.0, abs(lo), abs(hi))
        lo -= guard
        hi += guard
        dk = d[k]
        if dk == 0.0:
            if o[k] < lo or o[k] > hi:
                return False
            continue
        t1 = (lo - o[k]) / dk
        t2 = (hi - o[k]) / dk
        if t1 < t2:
            tn, tf = t1, t2
        else:
            tn, tf = t2, t1
        if tn > tnear:
            tnear = tn
        if tf < tfar:
            tfar = tf
        if tnear > tfar:
            return False
    return True


def _triangle_hit(face, o, d, tau, t_min, t_max):
    """Möller–Trumbore（双面、不背面剔除）。返回 (t,u,v) 或 None。"""
    a = _vlerp(face[0], face[3], tau)
    b = _vlerp(face[1], face[4], tau)
    c = _vlerp(face[2], face[5], tau)
    e1 = _vsub(b, a)
    e2 = _vsub(c, a)
    pvec = _vcross(d, e2)
    det = _vdot(e1, pvec)
    if det == 0.0:
        return None  # 退化面（零面积）或与射线平行
    inv_det = 1.0 / det
    tvec = _vsub(o, a)
    u = _vdot(tvec, pvec) * inv_det
    if u < -_BARY_EPS or u > 1.0 + _BARY_EPS:
        return None
    qvec = _vcross(tvec, e1)
    v = _vdot(d, qvec) * inv_det
    if v < -_BARY_EPS or u + v > 1.0 + _BARY_EPS:
        return None
    t = _vdot(e2, qvec) * inv_det
    if t < t_min or t > t_max:
        return None
    # 边界吸收：仅在 _BARY_EPS 内夹到 [0,1]
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if v < 0.0:
        v = 0.0
    if u + v > 1.0:
        v = 1.0 - u
    return t, u, v
