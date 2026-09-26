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
# 最近点独立参考：逐三角形投影（暴力 oracle，不调用被测核心）
# ---------------------------------------------------------------------------

def _seg_closest(p, a, b):
    """独立书写的点-线段最近点。返回 (s∈[0,1], 点, 距离平方)。"""
    ab = _sub(b, a)
    dd = _dot(ab, ab)
    if dd == 0.0:
        d = _sub(p, a)
        return 0.0, a, _dot(d, d)
    s = _dot(_sub(p, a), ab) / dd
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    q = (a[0] + s * ab[0], a[1] + s * ab[1], a[2] + s * ab[2])
    d = _sub(p, q)
    return s, q, _dot(d, d)


def _tri_closest(tri, p, tau):
    """独立书写的点-三角形投影。返回 (u, v, 距离平方)。

    与被测实现不同源：先求平面投影的重心坐标（面积法），落在面内即取之；
    否则取三条边线段最近点（顺序 ab、bc、ca，严格更小才替换）。
    零面积退化面自动只剩线段候选；完全共点退化为点。
    """
    a = _lerp(tri.a0, tri.a1, tau)
    b = _lerp(tri.b0, tri.b1, tau)
    c = _lerp(tri.c0, tri.c1, tau)
    ab = _sub(b, a)
    ac = _sub(c, a)
    best = None  # (d2, u, v)

    n = _cross(ab, ac)
    nn = _dot(n, n)
    if nn > 0.0:
        # 平面投影点 q，再用面积法求重心坐标
        dist = _dot(_sub(p, a), n) / nn
        q = (p[0] - dist * n[0], p[1] - dist * n[1], p[2] - dist * n[2])
        aq = _sub(q, a)
        d00 = _dot(ab, ab)
        d01 = _dot(ab, ac)
        d11 = _dot(ac, ac)
        d20 = _dot(aq, ab)
        d21 = _dot(aq, ac)
        denom = d00 * d11 - d01 * d01
        if denom > 0.0:
            u = (d11 * d20 - d01 * d21) / denom
            v = (d00 * d21 - d01 * d20) / denom
            w = 1.0 - u - v
            if u >= 0.0 and v >= 0.0 and w >= 0.0:
                d = _sub(p, q)
                best = (_dot(d, d), u, v)

    # 三条边线段候选（也覆盖退化面；bc 边 u=1-s、v=s，ca 边 v=1-s）
    for p0, p1, to_uv in (
        (a, b, lambda s: (s, 0.0)),
        (b, c, lambda s: (1.0 - s, s)),
        (c, a, lambda s: (0.0, 1.0 - s)),
    ):
        s, _q, d2 = _seg_closest(p, p0, p1)
        u, v = to_uv(s)
        if best is None or d2 < best[0]:
            best = (d2, u, v)
    d2, u, v = best
    return u, v, d2


def reference_closest_point(triangles, point, time):
    """暴力对全部三角形逐一投影。triangles 为原始 Triangle 列表。

    返回 dict：命中 {hit,face_id,u,v,dist_sq,point,triangles_tested}，
    空网格 {hit:False, triangles_tested:0}。
    决胜规则与被测契约一致：严格更近者胜；距离平方差在 16 ulp 相对容差
    内视为等距，face_id 较小者胜。
    """
    p = tuple(float(x) for x in point)
    tau = float(time)
    by_id = {}
    cands = []  # (d2,u,v,fid)
    tested = 0
    for i, tri in enumerate(triangles):
        fid = i if tri.face_id is None else tri.face_id
        by_id[fid] = tri
        tested += 1
        u, v, d2 = _tri_closest(tri, p, tau)
        cands.append((d2, u, v, fid))
    assert tested == len(triangles)
    if not cands:
        return {"hit": False, "triangles_tested": tested}

    d_star = min(c[0] for c in cands)
    in_band = [c for c in cands
               if c[0] - d_star <= _TIE_ULPS * _EPS
               * max(1.0, abs(c[0]), abs(d_star))]
    d2, u, v, fid = min(in_band, key=lambda c: c[3])
    tri = by_id[fid]
    a = _lerp(tri.a0, tri.a1, tau)
    b = _lerp(tri.b0, tri.b1, tau)
    c_ = _lerp(tri.c0, tri.c1, tau)
    w = 1.0 - u - v
    point_out = (w * a[0] + u * b[0] + v * c_[0],
                 w * a[1] + u * b[1] + v * c_[1],
                 w * a[2] + u * b[2] + v * c_[2])
    return {"hit": True, "face_id": fid, "u": u, "v": v, "dist_sq": d2,
            "point": point_out, "triangles_tested": tested}
