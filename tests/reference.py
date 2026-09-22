"""独立参考实现：对全部三角形逐一直接求交。

本模块**不导入 motion_bvh**，与被测核心走不同的计算路径：
  * 三角形相交用“平面相交 + 边向量投影求重心坐标”，而非核心的 Möller–Trumbore；
  * 不使用任何层次结构，线性扫描全部三角形；
  * 同距按 face_id 取小的规则在此独立实现一份。

仅供 tests/ 与 demo.py 比对，不是题库公开接口。
"""

TIE_EPS = float.fromhex("0x1.0p-45")
EDGE_EPS = float.fromhex("0x1.0p-48")


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _interpolate_vertices(triangle, time):
    v0, v1 = triangle
    s = 1.0 - time
    return tuple(
        tuple(v0[k][axis] * s + v1[k][axis] * time for axis in range(3)) for k in range(3)
    )


def _single_triangle(triangle, time, origin, direction, tmin, tmax):
    """返回 (t, u, v, w)（u/v/w 对应顶点 a/b/c）或 None。独立公式。"""
    a, b, c = _interpolate_vertices(triangle, time)
    e1 = _sub(b, a)
    e2 = _sub(c, a)

    # |e1 x e2|^2 == 0 即零面积（含塌缩点/线）。
    g11 = _dot(e1, e1)
    g22 = _dot(e2, e2)
    g12 = _dot(e1, e2)
    denom = g11 * g22 - g12 * g12
    if denom == 0.0:
        return None

    normal = _cross(e1, e2)
    ddn = _dot(direction, normal)
    if ddn == 0.0:
        return None  # 射线与平面平行。

    t = _dot(_sub(a, origin), normal) / ddn
    t_guard = EDGE_EPS * max(1.0, abs(t))
    if t < tmin - t_guard or t > tmax + t_guard:
        return None

    point = (
        origin[0] + t * direction[0],
        origin[1] + t * direction[1],
        origin[2] + t * direction[2],
    )
    svec = _sub(point, a)
    # 用边向量度量矩阵直接解重心坐标（与核心 MT 路径不同）。
    v = (g22 * _dot(svec, e1) - g12 * _dot(svec, e2)) / denom
    w = (g11 * _dot(svec, e2) - g12 * _dot(svec, e1)) / denom
    u = 1.0 - v - w
    if u < -EDGE_EPS or v < -EDGE_EPS or w < -EDGE_EPS:
        return None
    return t, u, v, w


def reference_intersect(triangles, origin, direction, time, tmin=0.0, tmax=float("inf")):
    """线性扫描全部三角形。triangles 为 ((v0三元组),(v1三元组)) 序列。

    返回 dict(t, face_id, barycentric) 或 None。
    """
    best = None
    for face_id, triangle in enumerate(triangles):
        hit = _single_triangle(triangle, time, origin, direction, tmin, tmax)
        if hit is None:
            continue
        t, u, v, w = hit
        if best is None:
            best = (t, face_id, u, v, w)
            continue
        band = TIE_EPS * max(1.0, abs(t), abs(best[0]))
        if t < best[0] - band:
            # 严格更近：另起同距簇。
            best = (t, face_id, u, v, w)
        elif t <= best[0] + band and face_id < best[1]:
            # 同距簇内：face_id 取小，簇距离取小（规则与被测核心一致，实现独立）。
            best = (min(t, best[0]), face_id, u, v, w)
    if best is None:
        return None
    t, face_id, u, v, w = best
    total = u + v + w
    return {
        "t": t,
        "face_id": face_id,
        "barycentric": (u / total, v / total, w / total),
    }
