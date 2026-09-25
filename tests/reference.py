"""测试用独立参考实现：对全部三角形逐一直接相交（暴力 oracle）。

本模块刻意不 import motion_bvh 的任何算法函数（build/intersect 及内部
遍历/求交均不调用），只共享数据输入形态（读取 Triangle/Ray 的字段）。
它用一份独立书写的 Möller–Trumbore 求交与同一套契约决胜规则，作为
BVH 结果的期望值来源。契约常量在此独立抄写一份，避免“从被测模块取
期望值”。
"""

from __future__ import annotations

import math
from fractions import Fraction

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
# 遮挡（阴影）查询的独立参考：与被测 packet_occluded 的共享调度遍历完全无关，
# 逐条射线 × 逐个三角形直接求交。
# ---------------------------------------------------------------------------

def reference_occluded(triangles, origin, direction, time,
                       t_min=0.0, t_max=math.inf, ignore_faces=()):
    """浮点暴力遮挡参考：半开区间 [t_min, t_max)，忽略集合只属本条射线。

    独立抄写一份半开区间 Möller–Trumbore（t==t_max 接触不算遮挡），
    不调用 motion_bvh 的任何函数。返回 bool。
    """
    o = tuple(origin)
    d = tuple(direction)
    tau = float(time)
    ignored = frozenset(ignore_faces or ())
    for i, tri in enumerate(triangles):
        fid = i if tri.face_id is None else tri.face_id
        if fid in ignored:
            continue
        a = _lerp(tri.a0, tri.a1, tau)
        b = _lerp(tri.b0, tri.b1, tau)
        c = _lerp(tri.c0, tri.c1, tau)
        e1 = _sub(b, a)
        e2 = _sub(c, a)
        p = _cross(d, e2)
        det = _dot(e1, p)
        if det == 0.0:
            continue
        inv = 1.0 / det
        tv = _sub(o, a)
        u = _dot(tv, p) * inv
        if u < -_BARY_EPS or u > 1.0 + _BARY_EPS:
            continue
        q = _cross(tv, e1)
        v = _dot(d, q) * inv
        if v < -_BARY_EPS or u + v > 1.0 + _BARY_EPS:
            continue
        t = _dot(e2, q) * inv
        if t < t_min or t >= t_max:  # 半开：t==t_max 不算遮挡
            continue
        return True
    return False


def reference_packet_occluded_exact(triangles, packet):
    """精确有理数（Fraction）逐面遮挡参考：零容差、零浮点。

    仅用于坐标/射线参数全为小整数的测试场景（精确 oracle，与被测实现的
    浮点 + 边界宽限口径互相独立）。packet 为
    (origin, direction, time, t_min, t_max, ignore_faces) 元组列表，
    元素均为 int/Fraction；time 只取 0 或 1（整数时刻）。
    返回 tuple[bool, ...]，按输入位置一一对应。
    """
    out = []
    for (o, d, tau, t0, t1, ignore) in packet:
        o = tuple(Fraction(x) for x in o)
        d = tuple(Fraction(x) for x in d)
        tau = Fraction(tau)
        t0 = Fraction(t0)
        t1 = Fraction(t1) if not math.isinf(float(t1)) else None
        ignored = frozenset(ignore or ())
        hit = False
        for i, tri in enumerate(triangles):
            fid = i if tri.face_id is None else tri.face_id
            if fid in ignored:
                continue
            a = tuple(Fraction(tri.a0[k]) + tau * (Fraction(tri.a1[k]) - Fraction(tri.a0[k]))
                      for k in range(3))
            b = tuple(Fraction(tri.b0[k]) + tau * (Fraction(tri.b1[k]) - Fraction(tri.b0[k]))
                      for k in range(3))
            c = tuple(Fraction(tri.c0[k]) + tau * (Fraction(tri.c1[k]) - Fraction(tri.c0[k]))
                      for k in range(3))
            e1 = tuple(b[k] - a[k] for k in range(3))
            e2 = tuple(c[k] - a[k] for k in range(3))
            p = (d[1] * e2[2] - d[2] * e2[1],
                 d[2] * e2[0] - d[0] * e2[2],
                 d[0] * e2[1] - d[1] * e2[0])
            det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2]
            if det == 0:
                continue
            tv = tuple(o[k] - a[k] for k in range(3))
            u = (tv[0] * p[0] + tv[1] * p[1] + tv[2] * p[2]) / det
            if u < 0 or u > 1:
                continue
            q = (tv[1] * e1[2] - tv[2] * e1[1],
                 tv[2] * e1[0] - tv[0] * e1[2],
                 tv[0] * e1[1] - tv[1] * e1[0])
            v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det
            if v < 0 or u + v > 1:
                continue
            t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det
            if t < t0:
                continue
            if t1 is not None and t >= t1:  # 半开：t==t_max 不算遮挡
                continue
            hit = True
            break
        out.append(hit)
    return tuple(out)
