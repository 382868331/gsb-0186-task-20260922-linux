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

实例层（构建后不变）：
  * Instance 引用一个已构建的 BVH（多个实例共享同一棵底层树，不复制面列表），
    带唯一非负 instance_id、固定可逆 3×3 线性矩阵 M 与平移 T（世界点 = M·局部+T），
    允许旋转、非均匀缩放、剪切、镜像（负行列式）。
  * build_scene(instances) -> SceneBVH：对实例的“整运动区间保守世界 AABB”
    （两端点 AABB 分别变换到世界后的并集）再建一层 8 桶 SAH 树。
  * intersect_scene 把世界射线变到实例局部（o'=M⁻¹(o−T)、d'=M⁻¹d，方向
    **不归一化**，故局部/世界射线参数 t 相同），遍历底层树；结果返回
    instance_id、原 face_id、世界命中点、原局部重心坐标。同距带沿用基线
    定义，带内先按 instance_id 再按 face_id 决胜。
  * 计数区分：scene_boxes_tested（场景盒）、underlying_boxes_tested（底层
    网格盒）、triangles_tested（实际三角形）。
  * 全部实例先验证再建场景：重复 ID、非有限数、奇异矩阵、逆矩阵非有限值
    都会整体拒绝，不产生部分场景。旧 build/intersect 不受影响。

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
    "Instance",
    "SceneIntersection",
    "SceneBVH",
    "build_scene",
    "intersect_scene",
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


@dataclass(frozen=True, slots=True)
class Instance:
    """一个固定实例：引用已构建的 BVH，外加实例自身的刚体般放置变换。

    bvh              被引用的底层 BVH；多个 Instance 可引用同一个 BVH，
                     共享底层树与面数据，不复制、不重新 build。
    instance_id      场景内唯一的非负 int（同距决胜的第一关键字）。
    matrix           可逆 3×3 线性部分，按行给出的 3 个长度 3 序列；
                     世界点 P = M·p + translation。允许旋转、非均匀缩放、
                     剪切、镜像（行列式可为负），但必须可逆（非奇异）。
    translation      长度 3 的平移向量，缺省为 (0,0,0)。
    变换在时间上固定；被引用网格仍按查询的 time 参数运动。
    """

    bvh: "BVH"
    instance_id: int
    matrix: Sequence[Sequence[float]]
    translation: Optional[Sequence[float]] = None


@dataclass(frozen=True, slots=True)
class SceneIntersection:
    """实例场景查询结果。未命中时 hit=False，三个计数字段仍有效。

    hit                       是否命中
    instance_id               命中实例标识；未命中为 None
    face_id                   该实例底层网格中的原 face_id；未命中为 None
    t                         射线参数（世界与局部一致：方向未归一化）
    u, v                      原局部网格的重心坐标（不变换回来）
    point                     世界坐标命中点
    scene_boxes_tested        场景层执行了射线测试的 AABB 数（含根）
    underlying_boxes_tested   所有实例底层树执行了盒测试的节点总数
    triangles_tested          所有实例底层树实际做三角形求交的面总数
    """

    hit: bool
    instance_id: Optional[int]
    face_id: Optional[int]
    t: float
    u: float
    v: float
    point: Optional[tuple]
    scene_boxes_tested: int
    underlying_boxes_tested: int
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


class _SNode:
    """场景层节点：叶节点持有若干实例记录，内部节点持有两个孩子。"""

    __slots__ = ("lo", "hi", "left", "right", "items")

    def __init__(self):
        self.lo = None
        self.hi = None
        self.left: Optional[_SNode] = None
        self.right: Optional[_SNode] = None
        # 叶节点：(instance_id, bvh, M 行元组, invM 行元组, T, centroid)
        self.items: Optional[tuple] = None


class SceneBVH:
    """构建后不变的实例层次。请通过 motion_bvh.build_scene 构造。

    场景节点存每个实例“整运动区间”的保守世界 AABB（底层 BVH 两端 AABB
    分别做仿射变换后的并集），该盒与查询时刻无关；实例被引用的底层 BVH
    在多个实例间共享，不复制面列表、不重新建树。
    """

    __slots__ = ("_root", "_count", "_shared_bvh")

    def __init__(self):
        raise TypeError("SceneBVH 不可直接构造，请使用 motion_bvh.build_scene()")

    @classmethod
    def _from_root(cls, root, count, shared_bvh):
        self = object.__new__(cls)
        self._root = root
        self._count = count
        # 仅用于“共享同一棵树”的自省/断言，不参与遍历
        self._shared_bvh = shared_bvh
        return self

    @property
    def instance_count(self) -> int:
        return self._count

    def intersect(
        self,
        ray: Ray,
        time: float,
        t_min: float = 0.0,
        t_max: float = math.inf,
    ) -> SceneIntersection:
        return _intersect_scene(self, ray, time, t_min, t_max)


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
# 实例场景：校验 + 保守盒 + 第二层 SAH + 实例化遍历
# ---------------------------------------------------------------------------

_SCENE_LEAF_SIZE = 4


def build_scene(instances: Sequence[Instance]) -> SceneBVH:
    """由 Instance 列表构建实例场景层次。

    全量校验通过后才建树：重复 instance_id、非有限数、奇异矩阵、逆矩阵
    含非有限值、实例世界包围盒计算溢出（非有限）都会整体拒绝，不产生
    部分场景。空列表合法（空场景，任何查询未命中）。
    """
    if not isinstance(instances, (list, tuple)):
        try:
            instances = list(instances)
        except TypeError as exc:
            raise TypeError("instances 必须是 Instance 的可迭代对象") from exc
    n = len(instances)

    # 单遍校验 + 物化（与 build 同一风格：只有全部通过且末尾才返回对象；
    # 失败调用不修改任何已存在的 BVH/场景，无外部可见的部分结果）。
    seen_ids = set()
    records = []
    shared_bvh = None
    for i, inst in enumerate(instances):
        where = f"instances[{i}]"
        if not isinstance(inst, Instance):
            raise TypeError(
                f"{where} 必须是 motion_bvh.Instance，得到 {type(inst).__name__}"
            )
        iid = _check_int(inst.instance_id, f"{where}.instance_id")
        if iid < 0:
            raise ValueError(f"{where}.instance_id 必须非负，得到 {iid}")
        if iid in seen_ids:
            raise ValueError(f"{where}.instance_id={iid} 与前面的实例重复")
        seen_ids.add(iid)
        if not isinstance(inst.bvh, BVH):
            raise TypeError(
                f"{where}.bvh 必须是 motion_bvh.BVH（请先用 build 构建），"
                f"得到 {type(inst.bvh).__name__}"
            )
        M = _check_matrix(inst.matrix, f"{where}.matrix")
        if inst.translation is None:
            T = (0.0, 0.0, 0.0)
        else:
            T = _check_point(inst.translation, f"{where}.translation")
        invM, det = _invert_matrix3(M)
        if det == 0.0:
            raise ValueError(
                f"{where}.matrix 奇异（行列式为 0），实例变换必须可逆"
            )
        for r in range(3):
            for c in range(3):
                if not math.isfinite(invM[r][c]):
                    # 有限输入 + 非零行列式在普通量级下不会发生；一旦发生
                    # （如乘积溢出致 inf/NaN）如实拒绝，不放大容差放行。
                    raise ValueError(
                        f"{where}.matrix 的逆矩阵含非有限值，拒绝该变换"
                    )
        lo, hi = _instance_world_box(inst.bvh, M, T)
        for k in range(3):
            if not (math.isfinite(lo[k]) and math.isfinite(hi[k])):
                raise ValueError(
                    f"{where} 的世界包围盒含非有限值（变换与网格坐标之积溢出），"
                    f"拒绝该实例"
                )
        centroid = tuple((lo[k] + hi[k]) * 0.5 for k in range(3))
        records.append((iid, inst.bvh, M, invM, T, lo, hi, centroid))
        if i == 0:
            shared_bvh = inst.bvh
        elif shared_bvh is not None and inst.bvh is not shared_bvh:
            shared_bvh = None

    # 校验全部通过，开始建树（不会再失败）
    root = _build_scene(records, list(range(n))) if n else None
    return SceneBVH._from_root(root, n, shared_bvh)


def intersect_scene(
    scene: SceneBVH,
    ray: Ray,
    time: float,
    t_min: float = 0.0,
    t_max: float = math.inf,
) -> SceneIntersection:
    """模块级入口，等价于 scene.intersect(ray, time, t_min, t_max)。"""
    return _intersect_scene(scene, ray, time, t_min, t_max)


def _check_matrix(m, where: str):
    """校验 3×3 矩阵：3 行，每行 3 个有限实数，bool 拒绝；返回行元组。"""
    if not isinstance(m, (tuple, list)):
        raise TypeError(f"{where} 必须是 3×3 序列（3 行，每行 3 个数）")
    if len(m) != 3:
        raise TypeError(f"{where} 必须恰好含 3 行，得到 {len(m)}")
    rows = []
    for r in range(3):
        row = m[r]
        if not isinstance(row, (tuple, list)):
            raise TypeError(f"{where}[{r}] 必须是长度为 3 的序列(tuple/list)")
        if len(row) != 3:
            raise TypeError(
                f"{where}[{r}] 必须恰好含 3 个分量，得到 {len(row)}"
            )
        rows.append((
            _check_real(row[0], f"{where}[{r}][0]"),
            _check_real(row[1], f"{where}[{r}][1]"),
            _check_real(row[2], f"{where}[{r}][2]"),
        ))
    return tuple(rows)


def _invert_matrix3(m):
    """闭式 3×3 求逆（余因子/行列式），返回 (inv 行元组, det)。

    输入全部有限。det==0 表示奇异，由调用方拒绝；inv 出现非有限值
    同样由调用方拒绝。不设置任何“接近奇异”的阈值——非零即接受，
    但非有限的逆绝不放行。
    """
    (a, b, c), (d, e, f), (g, h, i) = m
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    det = a * A + b * B + c * C
    if det == 0.0:
        return None, 0.0
    inv_det = 1.0 / det
    inv = (
        (A * inv_det,
         (c * h - b * i) * inv_det,
         (b * f - c * e) * inv_det),
        (B * inv_det,
         (a * i - c * g) * inv_det,
         (c * d - a * f) * inv_det),
        ((d * h - e * g) * inv_det,  # == C*inv_det，保持展开形式独立抄写
         (b * g - a * h) * inv_det,
         (a * e - b * d) * inv_det),
    )
    return inv, det


def _apply_affine(M, T, p):
    return (
        M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + T[0],
        M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + T[1],
        M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + T[2],
    )


def _transform_box(lo, hi, M, T):
    """轴对齐盒经仿射变换后的保守 AABB：变换 8 个角点取并集。"""
    box = _EMPTY_BOX
    for sx in (lo[0], hi[0]):
        for sy in (lo[1], hi[1]):
            for sz in (lo[2], hi[2]):
                box = _union_box(box, _apply_affine(M, T, (sx, sy, sz)))
    return box


def _union_pair(box_a, box_b):
    (lo0, hi0), (lo1, hi1) = box_a, box_b
    return (
        (min(lo0[0], lo1[0]), min(lo0[1], lo1[1]), min(lo0[2], lo1[2])),
        (max(hi0[0], hi1[0]), max(hi0[1], hi1[1]), max(hi0[2], hi1[2])),
    )


def _instance_world_box(bvh: BVH, M, T):
    """实例整运动区间的保守世界 AABB。

    取底层根节点 t=0、t=1 两端 AABB 分别仿射变换到世界后的并集。
    顶点在两端之间线性运动、仿射保持线性，任意 τ 的变换后顶点都在
    两端变换位置的凸包内，故被两端变换盒的并集包含（保守界，不要求最紧）。
    """
    root = bvh._root
    if root is None or root.lo0 is None:
        # 空底层 BVH：无几何可包围。退化为平移点的空场景不会被查询命中，
        # 给一个平移点的退化盒（后续遍历时该实例的底层树也无节点可测）。
        p = T
        return (p, p)
    box0 = _transform_box(root.lo0, root.hi0, M, T)
    box1 = _transform_box(root.lo1, root.hi1, M, T)
    return _union_pair(box0, box1)


def _build_scene(records, indices):
    """场景层建树：t=0 参考质心的 8 桶 SAH（复用思路，独立一份），
    退化时稳定中分；叶至多 _SCENE_LEAF_SIZE 个实例。"""
    node = _SNode()
    boxes = []
    cents = []
    for j in indices:
        lo, hi = records[j][5], records[j][6]
        boxes.append((lo, hi))
        cents.append(records[j][7])
        if node.lo is None:
            node.lo, node.hi = lo, hi
        else:
            node.lo = (min(node.lo[0], lo[0]),
                       min(node.lo[1], lo[1]), min(node.lo[2], lo[2]))
            node.hi = (max(node.hi[0], hi[0]),
                       max(node.hi[1], hi[1]), max(node.hi[2], hi[2]))

    n = len(indices)
    if n <= _SCENE_LEAF_SIZE:
        node.items = tuple(records[j] for j in indices)
        return node

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

    split = _scene_sah_split(indices, boxes, cents, axis, clo[axis], chi[axis])
    if split is None:
        split = _scene_mid_split(indices, cents, axis)
    left_idx, right_idx = split
    node.left = _build_scene(records, left_idx)
    node.right = _build_scene(records, right_idx)
    return node


def _scene_sah_split(indices, boxes, cents, axis, cmin, cmax):
    n = len(indices)
    if not (cmax > cmin):
        return None
    inv = 1.0 / (cmax - cmin)
    bin_count = [0] * BIN_COUNT
    bin_lo = [[math.inf] * 3 for _ in range(BIN_COUNT)]
    bin_hi = [[-math.inf] * 3 for _ in range(BIN_COUNT)]
    for j, idx in enumerate(indices):
        x = cents[j][axis]
        k = int(BIN_COUNT * (x - cmin) * inv)
        if k >= BIN_COUNT:
            k = BIN_COUNT - 1
        bin_count[k] += 1
        lo, hi = boxes[j]
        for q in range(3):
            if lo[q] < bin_lo[k][q]:
                bin_lo[k][q] = lo[q]
            if hi[q] > bin_hi[k][q]:
                bin_hi[k][q] = hi[q]

    prefix = [None] * (BIN_COUNT - 1)
    cnt = 0
    plo = [math.inf] * 3
    phi = [-math.inf] * 3
    for k in range(BIN_COUNT - 1):
        cnt += bin_count[k]
        for q in range(3):
            if bin_lo[k][q] < plo[q]:
                plo[q] = bin_lo[k][q]
            if bin_hi[k][q] > phi[q]:
                phi[q] = bin_hi[k][q]
        if cnt:
            prefix[k] = (cnt, _box_surface_area(plo, phi))

    suffix = [None] * (BIN_COUNT - 1)
    cnt = 0
    slo = [math.inf] * 3
    shi = [-math.inf] * 3
    for k in range(BIN_COUNT - 1, 0, -1):
        cnt += bin_count[k]
        for q in range(3):
            if bin_lo[k][q] < slo[q]:
                slo[q] = bin_lo[k][q]
            if bin_hi[k][q] > shi[q]:
                shi[q] = bin_hi[k][q]
        if cnt:
            suffix[k - 1] = (cnt, _box_surface_area(slo, shi))

    par_lo = [math.inf] * 3
    par_hi = [-math.inf] * 3
    for k in range(BIN_COUNT):
        for q in range(3):
            if bin_lo[k][q] < par_lo[q]:
                par_lo[q] = bin_lo[k][q]
            if bin_hi[k][q] > par_hi[q]:
                par_hi[q] = bin_hi[k][q]
    parent_sa = _box_surface_area(par_lo, par_hi)
    if parent_sa <= 0.0:
        return None

    best_cost = math.inf
    best_plane = -1
    for k in range(BIN_COUNT - 1):
        lp, rp = prefix[k], suffix[k]
        if lp is None or rp is None:
            continue
        cost = 1.0 + (lp[0] * lp[1] + rp[0] * rp[1]) / parent_sa
        if cost < best_cost:
            best_cost = cost
            best_plane = k
    if best_plane < 0:
        return None
    cut = cmin + (best_plane + 1) * (cmax - cmin) / BIN_COUNT
    left, right = [], []
    for j, idx in enumerate(indices):
        if cents[j][axis] < cut:
            left.append(idx)
        else:
            right.append(idx)
    if not left or not right:
        return None
    if best_cost >= float(n):
        return None
    return left, right


def _scene_mid_split(indices, cents, axis):
    order = sorted(
        range(len(indices)),
        key=lambda j: (cents[j][axis], cents[j][(axis + 1) % 3],
                       cents[j][(axis + 2) % 3], indices[j]),
    )
    half = len(order) // 2
    return ([indices[order[j]] for j in range(half)],
            [indices[order[j]] for j in range(half, len(order))])


def _scene_box_hit(lo, hi, o, d, t_min, t_max) -> bool:
    """Slab 法射线-场景静态 AABB。边界包含；零方向分量单独处理。

    场景盒是 8 角点并集构造的精确（在 binary64 直接运算意义下）保守盒，
    不做 lerp；为吸收角点变换与 min/max 的舍入，各轴仅外扩 4 ulp 的
    相对护栏（与底层盒 lerp 护栏同量级），不用它掩盖几何错误。
    """
    tnear, tfar = t_min, t_max
    for k in range(3):
        lk, hk = lo[k], hi[k]
        guard = _BOX_GUARD_ULPS * _EPS * max(1.0, abs(lk), abs(hk))
        lk -= guard
        hk += guard
        dk = d[k]
        if dk == 0.0:
            if o[k] < lk or o[k] > hk:
                return False
            continue
        t1 = (lk - o[k]) / dk
        t2 = (hk - o[k]) / dk
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


def _intersect_scene(scene: SceneBVH, ray: Ray, time, t_min, t_max) -> SceneIntersection:
    if not isinstance(scene, SceneBVH):
        raise TypeError("ray 查询目标必须是 motion_bvh.SceneBVH")
    if not isinstance(ray, Ray):
        raise TypeError("ray 必须是 motion_bvh.Ray")
    # 与 _intersect 相同的参数校验（time/t 窗/射线），在进入遍历前完成。
    tau = _check_real(time, "time")
    if not (_TIME_LO <= tau <= _TIME_HI):
        raise ValueError(f"time 必须在 [{_TIME_LO}, {_TIME_HI}] 内，得到 {tau}")
    t0 = _check_real(t_min, "t_min")
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
    world_o = _check_point(ray.origin, "ray.origin")
    if not isinstance(ray.direction, (tuple, list)) or len(ray.direction) != 3:
        raise TypeError("ray.direction 必须是长度为 3 的序列")
    world_d = _check_point(ray.direction, "ray.direction")
    if world_d == (0.0, 0.0, 0.0):
        raise ValueError("ray.direction 不能为零向量")

    root = scene._root
    if root is None or root.lo is None:
        return SceneIntersection(
            False, None, None, math.nan, math.nan, math.nan, None, 0, 0, 0)

    scene_boxes = 0
    under_boxes = 0
    tris_tested = 0

    def tie_tol(a, b):
        return _TIE_ULPS * _EPS * max(1.0, abs(a), abs(b))

    # 与基线相同的两阶段决胜：全局严格最小 t*，同距带内先 instance_id
    # 再 face_id。win = (t, u, v, instance_id, face_id, rec, face_idx)
    abs_best = t1
    win = None

    def consider_instance(rec):
        nonlocal under_boxes, tris_tested, abs_best, win
        iid, bvh, M, invM, T = rec[0], rec[1], rec[2], rec[3], rec[4]
        # 世界射线 -> 实例局部。方向不做归一化：
        # P_world = o + t·d = T + M·p_local ⇒ p_local = M⁻¹(o−T) + t·M⁻¹d，
        # 故局部射线参数与世界参数完全相同。
        ox = world_o[0] - T[0]
        oy = world_o[1] - T[1]
        oz = world_o[2] - T[2]
        lo = (
            invM[0][0] * ox + invM[0][1] * oy + invM[0][2] * oz,
            invM[1][0] * ox + invM[1][1] * oy + invM[1][2] * oz,
            invM[2][0] * ox + invM[2][1] * oy + invM[2][2] * oz,
        )
        ld = (
            invM[0][0] * world_d[0] + invM[0][1] * world_d[1]
            + invM[0][2] * world_d[2],
            invM[1][0] * world_d[0] + invM[1][1] * world_d[1]
            + invM[1][2] * world_d[2],
            invM[2][0] * world_d[0] + invM[2][1] * world_d[1]
            + invM[2][2] * world_d[2],
        )
        # invM 可逆 ⇒ 非零向量映为非零向量（构建时已校验），ld 必非零。
        def window_now():
            if math.isinf(abs_best):
                return abs_best
            return abs_best + (_TIE_ULPS + 2) * _EPS * max(1.0, abs(abs_best))

        brow = bvh._root
        if brow is None or brow.lo0 is None:
            return  # 空底层树：无盒可测、无面可测
        stack = [brow]
        while stack:
            node = stack.pop()
            under_boxes += 1
            if not _box_hit(node, lo, ld, tau, t0, window_now()):
                continue
            if node.faces is not None:
                for idx in node.faces:
                    tris_tested += 1
                    face = bvh._faces[idx]
                    hit = _triangle_hit(face, lo, ld, tau, t0, window_now())
                    if hit is None:
                        continue
                    ht, hu, hv = hit
                    if ht > t1:  # 用户窗口硬边界
                        continue
                    fid = bvh._face_ids[idx]
                    key = (iid, fid)
                    if win is None:
                        abs_best = ht
                        win = (ht, hu, hv, iid, fid, rec, idx)
                    elif ht < abs_best:
                        abs_best = ht
                        band = tie_tol(win[0], ht)
                        if win[0] > ht + band or key < (win[3], win[4]):
                            win = (ht, hu, hv, iid, fid, rec, idx)
                    else:
                        band = tie_tol(ht, abs_best)
                        if abs(ht - abs_best) <= band and key < (win[3], win[4]):
                            win = (ht, hu, hv, iid, fid, rec, idx)
            else:
                stack.append(node.right)
                stack.append(node.left)

    stack = [root]
    while stack:
        node = stack.pop()
        scene_boxes += 1
        if math.isinf(abs_best):
            window = abs_best
        else:
            window = abs_best + (_TIE_ULPS + 2) * _EPS * max(1.0, abs(abs_best))
        if not _scene_box_hit(node.lo, node.hi, world_o, world_d, t0, window):
            continue
        if node.items is not None:
            # 叶节点：对其中每个实例先测该实例自己的整区间世界盒
            # （计入场景盒测试），通过后才进入其底层树。
            for rec in node.items:
                scene_boxes += 1
                if math.isinf(abs_best):
                    window = abs_best
                else:
                    window = abs_best + (_TIE_ULPS + 2) * _EPS * max(
                        1.0, abs(abs_best))
                if not _scene_box_hit(rec[5], rec[6], world_o, world_d,
                                      t0, window):
                    continue
                consider_instance(rec)
        else:
            stack.append(node.right)
            stack.append(node.left)

    if win is None:
        return SceneIntersection(
            False, None, None, math.nan, math.nan, math.nan, None,
            scene_boxes, under_boxes, tris_tested,
        )

    best_t, best_u, best_v, best_iid, best_fid, rec, face_idx = win
    bvh = rec[1]
    M, T = rec[2], rec[4]
    face = bvh._faces[face_idx]
    # 局部命中点：用局部插值顶点 + 原局部重心坐标组合（与基线口径一致）。
    a = _vlerp(face[0], face[3], tau)
    b = _vlerp(face[1], face[4], tau)
    c = _vlerp(face[2], face[5], tau)
    w = 1.0 - best_u - best_v
    lx = w * a[0] + best_u * b[0] + best_v * c[0]
    ly = w * a[1] + best_u * b[1] + best_v * c[1]
    lz = w * a[2] + best_u * b[2] + best_v * c[2]
    # 世界命中点 = M·局部点 + T。
    point = _apply_affine(M, T, (lx, ly, lz))
    return SceneIntersection(
        True, best_iid, best_fid, best_t, best_u, best_v, point,
        scene_boxes, under_boxes, tris_tested,
    )


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
