"""测试用独立参考实现：对全部三角形逐一直接相交（暴力 oracle）。

本模块刻意不 import motion_bvh 的任何算法函数（build/intersect 及内部
遍历/求交均不调用），只共享数据输入形态（读取 Triangle/Ray 的字段）。
它用一份独立书写的 Möller–Trumbore 求交与同一套契约决胜规则，作为
BVH 结果的期望值来源。契约常量在此独立抄写一份，避免“从被测模块取
期望值”。
"""

from __future__ import annotations

import math

_EPS = 2.220446049250313e-16
_BARY_EPS = 1e-12
_TIE_ULPS = 16


def _lerp(a, b, t):
    return (a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _tri_intersect(tri, o, d, tau, t_min, t_max):
    """独立书写的 Möller–Trumbore；返回 (face_id,t,u,v) 或 None。"""
    a = _lerp(tri.a0, tri.a1, tau)
    b = _lerp(tri.b0, tri.b1, tau)
    c = _lerp(tri.c0, tri.c1, tau)
    e1 = _sub(b, a)
    e2 = _sub(c, a)
    p = _cross(d, e2)
    det = _dot(e1, p)
    if det == 0.0:
        return None
    inv = 1.0 / det
    tv = _sub(o, a)
    u = _dot(tv, p) * inv
    if u < -_BARY_EPS or u > 1.0 + _BARY_EPS:
        return None
    q = _cross(tv, e1)
    v = _dot(d, q) * inv
    if v < -_BARY_EPS or u + v > 1.0 + _BARY_EPS:
        return None
    t = _dot(e2, q) * inv
    if t < t_min or t > t_max:
        return None
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if v < 0.0:
        v = 0.0
    if u + v > 1.0:
        v = 1.0 - u
    return t, u, v


def reference_intersect(triangles, ray, time, t_min=0.0, t_max=math.inf):
    """暴力遍历所有三角形。triangles 为原始 Triangle 列表（含 face_id 语义）。

    返回 dict：命中 {hit,face_id,t,u,v,point}，未命中 {hit:False}。
    决胜规则与被测契约一致：严格更近者胜；|Δt| 在 16 ulp 相对容差内视为
    同距，face_id 较小者胜。
    """
    o = tuple(ray.origin)
    d = tuple(ray.direction)
    tau = float(time)
    by_id = {}
    hits = []  # (t,u,v,fid)
    tested = 0
    for i, tri in enumerate(triangles):
        fid = i if tri.face_id is None else tri.face_id
        by_id[fid] = tri
        tested += 1
        r = _tri_intersect(tri, o, d, tau, t_min, t_max)
        if r is not None:
            t, u, v = r
            hits.append((t, u, v, fid))
    assert tested == len(triangles)
    if not hits:
        return {"hit": False, "triangles_tested": tested}

    # 两阶段、与遍历顺序无关：先取严格最小 t*，再在 |t-t*| ≤ 16 ulp 带内
    # 取最小 face_id（独立于被测代码重新实现同一契约）。
    t_star = min(h[0] for h in hits)
    in_band = [h for h in hits
               if abs(h[0] - t_star) <= _TIE_ULPS * _EPS
               * max(1.0, abs(h[0]), abs(t_star))]
    t, u, v, fid = min(in_band, key=lambda h: h[3])
    tri = by_id[fid]
    a = _lerp(tri.a0, tri.a1, tau)
    b = _lerp(tri.b0, tri.b1, tau)
    c = _lerp(tri.c0, tri.c1, tau)
    w = 1.0 - u - v
    point = (w * a[0] + u * b[0] + v * c[0],
             w * a[1] + u * b[1] + v * c[1],
             w * a[2] + u * b[2] + v * c[2])
    return {"hit": True, "face_id": fid, "t": t, "u": u, "v": v,
            "point": point, "triangles_tested": tested}


def _tri_by_id(triangles, fid):
    for j, tr in enumerate(triangles):
        if (tr.face_id if tr.face_id is not None else j) == fid:
            return tr
    raise LookupError(f"reference 内部错误：找不到 face_id={fid}")


# ---------------------------------------------------------------------------
# 最近点独立参考：逐三角形投影（暴力 oracle，不调用 motion_bvh）
# ---------------------------------------------------------------------------

def _seg_closest(p, x, y):
    """线段 xy 上距 p 最近点参数 s（独立书写）。零长线段返回 0。"""
    ex = y[0] - x[0]
    ey = y[1] - x[1]
    ez = y[2] - x[2]
    ll = ex * ex + ey * ey + ez * ez
    if ll == 0.0:
        return 0.0
    s = ((p[0] - x[0]) * ex + (p[1] - x[1]) * ey + (p[2] - x[2]) * ez) / ll
    return min(1.0, max(0.0, s))


def _project_tri(p, a, b, c):
    """独立书写的点-三角形投影，返回 (dist_sq, u, v)，
    P = (1-u-v)*a + u*b + v*c。覆盖面内、边上、顶点与退化（线段/点）面。

    实现策略（与被测模块不同的写法）：非退化面先求平面投影的重心坐标，
    若落在面外则退化为三条边线段投影取最近；退化面直接按边处理。
    """
    e1 = _sub(b, a)
    e2 = _sub(c, a)
    n = _cross(e1, e2)
    nn = _dot(n, n)
    if nn > 0.0:
        # 平面投影点的重心坐标（Cramer 法）
        ap = _sub(p, a)
        d11 = _dot(e1, e1)
        d12 = _dot(e1, e2)
        d22 = _dot(e2, e2)
        rhs1 = _dot(ap, e1)
        rhs2 = _dot(ap, e2)
        det = d11 * d22 - d12 * d12
        u = (d22 * rhs1 - d12 * rhs2) / det
        v = (d11 * rhs2 - d12 * rhs1) / det
        if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
            q = (a[0] + u * e1[0] + v * e2[0],
                 a[1] + u * e1[1] + v * e2[1],
                 a[2] + u * e1[2] + v * e2[2])
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            dz = p[2] - q[2]
            return dx * dx + dy * dy + dz * dz, u, v
    # 面外投影或退化面：三条边线段取最近（AB、BC、CA 固定顺序，严格更小才换）
    best = None
    for (x, y, fu) in (
        (a, b, lambda s: (s, 0.0)),
        (b, c, lambda s: (1.0 - s, s)),
        (c, a, lambda s: (0.0, 1.0 - s)),
    ):
        s = _seg_closest(p, x, y)
        q = (x[0] + s * (y[0] - x[0]),
             x[1] + s * (y[1] - x[1]),
             x[2] + s * (y[2] - x[2]))
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        dz = p[2] - q[2]
        d2 = dx * dx + dy * dy + dz * dz
        if best is None or d2 < best[0]:
            u, v = fu(s)
            best = (d2, u, v)
    return best


def reference_closest_point(triangles, point, time):
    """暴力对全部三角形逐一投影。返回 dict：
    命中 {hit,face_id,dist_sq,u,v,point,triangles_tested}，空网格 {hit:False}。

    决胜规则与契约一致：严格更近者胜；距离平方之差在 16 ulp 相对容差内
    视为等距，face_id 较小者胜（两阶段，与遍历顺序无关）。
    """
    p = tuple(float(x) for x in point)
    tau = float(time)
    cands = []  # (dist_sq, u, v, fid)
    by_id = {}
    tested = 0
    for i, tri in enumerate(triangles):
        fid = i if tri.face_id is None else tri.face_id
        by_id[fid] = tri
        tested += 1
        a = _lerp(tri.a0, tri.a1, tau)
        b = _lerp(tri.b0, tri.b1, tau)
        c = _lerp(tri.c0, tri.c1, tau)
        d2, u, v = _project_tri(p, a, b, c)
        cands.append((d2, u, v, fid))
    if not cands:
        return {"hit": False, "triangles_tested": tested}
    d_star = min(c[0] for c in cands)
    in_band = [c for c in cands
               if abs(c[0] - d_star) <= _TIE_ULPS * _EPS
               * max(1.0, abs(c[0]), abs(d_star))]
    d2, u, v, fid = min(in_band, key=lambda c: c[3])
    tri = by_id[fid]
    a = _lerp(tri.a0, tri.a1, tau)
    b = _lerp(tri.b0, tri.b1, tau)
    c = _lerp(tri.c0, tri.c1, tau)
    w = 1.0 - u - v
    pt = (w * a[0] + u * b[0] + v * c[0],
          w * a[1] + u * b[1] + v * c[1],
          w * a[2] + u * b[2] + v * c[2])
    return {"hit": True, "face_id": fid, "dist_sq": d2, "u": u, "v": v,
            "point": pt, "triangles_tested": tested}
