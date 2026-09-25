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
# 实例场景的独立参考：把每个实例的三角形在给定 τ 显式变换到世界坐标，
# 然后逐面 Möller–Trumbore。不 import motion_bvh，也不做任何“场景树/实例
# 遍历”——它只读取构造测试时本来就有的局部 Triangle 数据与实例变换。
# ---------------------------------------------------------------------------


def _apply(M, T, p):
    return (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + T[0],
            M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + T[1],
            M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + T[2])


def _world_triangle(tri, tau, M, T):
    """局部运动三角形在 τ 时刻的顶点，显式仿射变换到世界。"""
    a = _apply(M, T, _lerp(tri.a0, tri.a1, tau))
    b = _apply(M, T, _lerp(tri.b0, tri.b1, tau))
    c = _apply(M, T, _lerp(tri.c0, tri.c1, tau))
    return a, b, c


def _world_tri_intersect(a, b, c, o, d, t_min, t_max):
    """独立书写的世界空间 Möller–Trumbore；返回 (t,u,v) 或 None。"""
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


def reference_intersect_scene(instance_meshes, ray, time,
                              t_min=0.0, t_max=math.inf):
    """实例场景暴力 oracle。

    instance_meshes 为 [(instance_id, triangles, matrix, translation), ...]，
    triangles 是该实例 build 时使用的原始局部 Triangle 列表（期望值由测试
    直接持有，不从被测场景对象取）。每个三角形在 τ 时刻被显式变换到世界
    坐标后逐面求交；同距带沿用基线定义，带内决胜键为
    (instance_id, face_id)。

    返回 dict：命中 {hit,instance_id,face_id,t,u,v,point}，未命中
    {hit:False}；另含 scene_instances 计数（不做任何盒测试，仅供测试断言）。
    """
    o = tuple(ray.origin)
    d = tuple(ray.direction)
    tau = float(time)
    hits = []  # (t, u, v, instance_id, face_id, (a,b,c))
    tested = 0
    for iid, triangles, M, T in instance_meshes:
        for j, tri in enumerate(triangles):
            fid = j if tri.face_id is None else tri.face_id
            a, b, c = _world_triangle(tri, tau, M, T)
            tested += 1
            r = _world_tri_intersect(a, b, c, o, d, t_min, t_max)
            if r is not None:
                t, u, v = r
                hits.append((t, u, v, iid, fid, (a, b, c)))
    if not hits:
        return {"hit": False, "triangles_tested": tested,
                "instances": len(instance_meshes)}

    t_star = min(h[0] for h in hits)
    in_band = [h for h in hits
               if abs(h[0] - t_star) <= _TIE_ULPS * _EPS
               * max(1.0, abs(h[0]), abs(t_star))]
    # 决胜：先 instance_id，再 face_id。
    t, u, v, iid, fid, (a, b, c) = min(
        in_band, key=lambda h: (h[3], h[4]))
    w = 1.0 - u - v
    point = (w * a[0] + u * b[0] + v * c[0],
             w * a[1] + u * b[1] + v * c[1],
             w * a[2] + u * b[2] + v * c[2])
    return {"hit": True, "instance_id": iid, "face_id": fid,
            "t": t, "u": u, "v": v, "point": point,
            "triangles_tested": tested, "instances": len(instance_meshes)}
