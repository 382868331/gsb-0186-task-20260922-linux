"""motion_bvh 必要测试。

期望值来源：tests/reference.py 中独立书写的逐三角形暴力求交（不调用
motion_bvh 的 build/intersect 或任何内部算法）。往返一致性仅作补充。
运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import motion_bvh as mb
from motion_bvh import Triangle, Ray, build, intersect
from reference import reference_intersect


# ---------------------------------------------------------------------------
# 确定性数据生成
# ---------------------------------------------------------------------------

def _rand_tri(rng, motion_scale=2.0):
    def pt():
        return (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-4, 4))

    a0, b0, c0 = pt(), pt(), pt()
    mv = []
    for _ in range(3):
        mv.append((rng.uniform(-motion_scale, motion_scale),
                   rng.uniform(-motion_scale, motion_scale),
                   rng.uniform(-motion_scale, motion_scale)))
    a1 = tuple(a0[k] + mv[0][k] for k in range(3))
    b1 = tuple(b0[k] + mv[1][k] for k in range(3))
    c1 = tuple(c0[k] + mv[2][k] for k in range(3))
    return Triangle(a0, b0, c0, a1, b1, c1)


def _at(tri, which, tau):
    p0 = getattr(tri, which + "0")
    p1 = getattr(tri, which + "1")
    return tuple(p0[k] + tau * (p1[k] - p0[k]) for k in range(3))


def _centroid_at(tri, tau):
    a = _at(tri, "a", tau)
    b = _at(tri, "b", tau)
    c = _at(tri, "c", tau)
    return tuple((a[k] + b[k] + c[k]) / 3.0 for k in range(3))


def _aimed_ray(rng, tris, tau):
    """构造一条大致瞄准某三角形 τ 时刻质心的射线（含偏移，制造命中/擦边）。"""
    tri = rng.choice(tris)
    target = _centroid_at(tri, tau)
    angle = rng.random() * 2 * math.pi
    radius = rng.uniform(6.0, 12.0)
    origin = (target[0] + radius * math.cos(angle),
              target[1] + radius * math.sin(angle),
              target[2] + rng.uniform(4.0, 10.0))
    d = tuple(target[k] - origin[k] for k in range(3))
    # 偶尔加横向扰动，产生擦边/未命中
    if rng.random() < 0.35:
        j = rng.randrange(3)
        d = list(d)
        d[j] += rng.uniform(-1.5, 1.5)
        d = tuple(d)
    return Ray(origin, d)


def _wild_ray(rng):
    return Ray(
        tuple(rng.uniform(-20, 20) for _ in range(3)),
        tuple(rng.uniform(-1, 1) for _ in range(3)),
    )


def _assert_ref_agreement(test, bvh, tris, ray, tau, t_min=0.0, t_max=math.inf,
                          t_tol=1e-9, uv_tol=1e-9):
    got = intersect(bvh, ray, tau, t_min, t_max)
    exp = reference_intersect(tris, ray, tau, t_min, t_max)
    test.assertIs(got.hit, exp["hit"],
                  f"命中判定不一致 τ={tau} ray={ray}")
    if not exp["hit"]:
        return got
    test.assertEqual(got.face_id, exp["face_id"], "face_id / 同距决胜不一致")
    scale = max(1.0, abs(exp["t"]))
    test.assertLessEqual(abs(got.t - exp["t"]), t_tol * scale,
                         f"t 不一致: {got.t} vs {exp['t']}")
    test.assertLessEqual(abs(got.u - exp["u"]), uv_tol, "u 不一致")
    test.assertLessEqual(abs(got.v - exp["v"]), uv_tol, "v 不一致")
    for k in range(3):
        test.assertLessEqual(abs(got.point[k] - exp["point"][k]), 1e-8,
                             "命中点不一致")
    return got


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

class TestReferenceAgreement(unittest.TestCase):
    """与独立暴力参考全面比对（运动随机场景）。"""

    def test_fuzz_moving_scenes(self):
        rng = random.Random(20260922)
        total_rays = 0
        for scene in range(6):
            n = (0, 7, 13, 40, 63, 300)[scene]
            tris = [_rand_tri(rng) for _ in range(n)]
            bvh = build(tris)
            self.assertEqual(bvh.triangle_count, n)
            for _ in range(60 if n < 300 else 40):
                tau = rng.choice((0.0, 0.17, 0.5, 0.83, 1.0))
                if n and rng.random() < 0.7:
                    ray = _aimed_ray(rng, tris, tau)
                else:
                    ray = _wild_ray(rng)
                _assert_ref_agreement(self, bvh, tris, ray, tau)
                total_rays += 1
        self.assertGreater(total_rays, 200)

    def test_custom_face_ids_roundtrip(self):
        rng = random.Random(42)
        tris = [_rand_tri(rng) for _ in range(30)]
        ids = list(range(100, 130))
        rng.shuffle(ids)
        tris = [Triangle(t.a0, t.b0, t.c0, t.a1, t.b1, t.c1, face_id=fid)
                for t, fid in zip(tris, ids)]
        bvh = build(tris)
        for _ in range(50):
            tau = rng.choice((0.0, 0.5, 1.0))
            _assert_ref_agreement(self, bvh, tris, _aimed_ray(rng, tris, tau), tau)


class TestMotionSemantics(unittest.TestCase):
    def _moving_tri(self):
        # t=0 中心 x=0，t=1 中心 x=10，平面 z=5
        a0, b0, c0 = (-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0)
        a1 = (a0[0] + 10, a0[1], a0[2])
        b1 = (b0[0] + 10, b0[1], b0[2])
        c1 = (c0[0] + 10, c0[1], c0[2])
        return Triangle(a0, b0, c0, a1, b1, c1)

    def test_hit_depends_on_time(self):
        tri = self._moving_tri()
        bvh = build([tri])
        # τ=0：从 z 轴正向射向 (0,0,5)
        r0 = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertTrue(intersect(bvh, r0, 0.0).hit)
        # τ=1：三角形已移到 x=10，同射线未命中
        self.assertFalse(intersect(bvh, r0, 1.0).hit)
        # τ=0.5：瞄准插值位置 x=5
        r_half = Ray((0.0, 0.0, 0.0), (5.0, 0.0, 5.0))
        res = intersect(bvh, r_half, 0.5)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 1.0, places=10)
        # 参考一致性（端点与中间时刻）
        for tau, ray in ((0.0, r0), (1.0, r0), (0.5, r_half)):
            _assert_ref_agreement(self, bvh, [tri], ray, tau)

    def test_endpoint_times_exact(self):
        tri = self._moving_tri()
        bvh = build([tri])
        # time=1 精确取 p1：瞄准 (10,0,5)
        r1 = Ray((10.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect(bvh, r1, 1.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.point[0], 10.0, places=10)
        self.assertAlmostEqual(res.point[2], 5.0, places=10)


class TestBarycentric(unittest.TestCase):
    def test_known_barycentrics(self):
        tri = Triangle.stationary(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        bvh = build([tri])
        ray = Ray((0.2, 0.3, -1.0), (0.0, 0.0, 1.0))
        res = intersect(bvh, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 1.0, places=12)
        self.assertAlmostEqual(res.u, 0.2, places=12)
        self.assertAlmostEqual(res.v, 0.3, places=12)
        self.assertAlmostEqual(res.point[0], 0.2, places=12)
        self.assertAlmostEqual(res.point[1], 0.3, places=12)

    def test_triangle_edges_and_vertices(self):
        # 三个点对应 u=0、v=0、u+v=1 三种边界
        tri = Triangle.stationary(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        bvh = build([tri])
        for (x, y), (eu, ev) in [
            ((0.0, 0.0), (0.0, 0.0)),       # 顶点 a
            ((1.0, 0.0), (1.0, 0.0)),       # 顶点 b
            ((0.0, 1.0), (0.0, 1.0)),       # 顶点 c
            ((0.5, 0.0), (0.5, 0.0)),       # 边 ab
            ((0.0, 0.5), (0.0, 0.5)),       # 边 ac
            ((0.5, 0.5), (0.5, 0.5)),       # 边 bc
        ]:
            ray = Ray((x, y, -1.0), (0.0, 0.0, 1.0))
            res = intersect(bvh, ray, 0.0)
            self.assertTrue(res.hit, f"边界点 ({x},{y}) 应命中")
            self.assertAlmostEqual(res.u, eu, places=10)
            self.assertAlmostEqual(res.v, ev, places=10)


class TestBoxEdgeTraversal(unittest.TestCase):
    def test_ray_on_box_corner_hits_vertex(self):
        # 三角形顶点 (0,0,5) 恰为包围盒角点；射线沿盒边 x=0,y=0 行进
        tri = Triangle.stationary(
            (0.0, 0.0, 5.0), (1.0, 0.0, 5.0), (0.0, 1.0, 6.0))
        bvh = build([tri])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect(bvh, ray, 0.0)
        self.assertTrue(res.hit, "贴包围盒边的射线必须进入遍历并命中角点")
        self.assertAlmostEqual(res.t, 5.0, places=10)
        _assert_ref_agreement(self, bvh, [tri], ray, 0.0)

    def test_ray_along_box_edge_missing_geometry_still_enters(self):
        # 射线贴着包围盒边行进、但不命中任何面：必须进入叶节点测试并
        # 证明无解，而不是在盒相交处被错误剔除。
        # 两个三角形合盒 x[0,1.5] y[-0.5,1.5] z[5,6]；射线 x=0,y=0 穿盒而过，
        # 但在 z=5 处落在 tri1 的 y 范围外，z=6 处落在 tri2 的 x 范围外。
        tri1 = Triangle.stationary(
            (0.0, 0.5, 5.0), (1.0, 0.5, 5.0), (0.0, 1.5, 5.0))
        tri2 = Triangle.stationary(
            (0.5, -0.5, 6.0), (1.5, -0.5, 6.0), (0.5, 0.5, 6.0))
        bvh = build([tri1, tri2])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect(bvh, ray, 0.0)
        self.assertFalse(res.hit)
        self.assertGreaterEqual(res.triangles_tested, 1)
        _assert_ref_agreement(self, bvh, [tri1, tri2], ray, 0.0)

    def test_ray_clearly_outside_is_pruned(self):
        tri = Triangle.stationary(
            (1.0, 1.0, 5.0), (2.0, 1.0, 5.0), (1.0, 2.0, 5.0))
        bvh = build([tri])
        ray = Ray((-10.0, 1.5, 0.0), (0.0, 0.0, 1.0))
        res = intersect(bvh, ray, 0.0)
        self.assertFalse(res.hit)
        self.assertEqual(res.triangles_tested, 0)


class TestDegenerate(unittest.TestCase):
    def test_zero_area_triangles_never_hit(self):
        # 点三角形与共线三角形（含运动）
        degen = [
            Triangle.stationary((0, 0, 5), (0, 0, 5), (0, 0, 5)),
            Triangle.stationary((-1, 0, 5), (0, 0, 5), (1, 0, 5)),
            Triangle((-1, 0, 5), (0, 0, 5), (1, 0, 5),
                     (-1, 2, 5), (0, 2, 5), (1, 2, 5)),
        ]
        bvh = build(degen)
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        for tau in (0.0, 0.5, 1.0):
            res = intersect(bvh, ray, tau)
            self.assertFalse(res.hit)
            _assert_ref_agreement(self, bvh, degen, ray, tau)

    def test_degenerate_mixed_with_normal(self):
        rng = random.Random(7)
        tris = [_rand_tri(rng) for _ in range(20)]
        tris.append(Triangle.stationary((0, 0, 5), (0, 0, 5), (0, 0, 5)))
        tris.append(Triangle.stationary((-3.0, -3.0, 2.0),
                                        (3.0, -3.0, 2.0),
                                        (0.0, 3.0, 2.0)))
        bvh = build(tris)
        for tau in (0.0, 0.5, 1.0):
            for _ in range(30):
                ray = _aimed_ray(rng, tris, tau)
                _assert_ref_agreement(self, bvh, tris, ray, tau)

    def test_shared_centroid_uses_stable_midpoint_split(self):
        # 所有三角形质心相同（质心包围盒全退化）→ 稳定中分兜底
        tris = []
        for i in range(13):
            s = 0.1 * (i + 1)
            tris.append(Triangle.stationary(
                (-s, -s, 0.0), (s, -s, 0.0), (0.0, s, 0.0)))
        bvh = build(tris)  # 不得因无法分割而失败/违反叶上限

        def leaves(node):
            if node.faces is not None:
                return [node.faces]
            return leaves(node.left) + leaves(node.right)

        ls = leaves(bvh._root)
        self.assertTrue(all(len(f) <= mb.LEAF_SIZE for f in ls))
        rng = random.Random(11)
        for _ in range(40):
            _assert_ref_agreement(
                self, bvh, tris,
                Ray((rng.uniform(-3, 3), rng.uniform(-3, 3), -5.0),
                    (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 1.0)),
                0.0)


class TestSameDistanceTie(unittest.TestCase):
    def _coplanar_pair(self, id_a, id_b):
        # 两个三角形同平面 z=1，均覆盖射线穿过的 (0,0)
        a = Triangle((-2.0, -2.0, 1.0), (2.0, -2.0, 1.0), (0.0, 2.0, 1.0),
                     (-2.0, -2.0, 1.0), (2.0, -2.0, 1.0), (0.0, 2.0, 1.0),
                     face_id=id_a)
        b = Triangle((-2.0, 2.0, 1.0), (2.0, 2.0, 1.0), (0.0, -2.0, 1.0),
                     (-2.0, 2.0, 1.0), (2.0, 2.0, 1.0), (0.0, -2.0, 1.0),
                     face_id=id_b)
        return a, b

    def test_tie_lower_face_id_wins_even_if_listed_first(self):
        a, b = self._coplanar_pair(5, 2)  # 先列 id=5，较小 id=2 应胜
        bvh = build([a, b])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect(bvh, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertEqual(res.face_id, 2)
        self.assertAlmostEqual(res.t, 1.0, places=12)
        _assert_ref_agreement(self, bvh, [a, b], ray, 0.0)

    def test_tie_default_ids_earlier_wins(self):
        a, b = self._coplanar_pair(0, 1)
        a = Triangle(a.a0, a.b0, a.c0, a.a1, a.b1, a.c1)
        b = Triangle(b.a0, b.b0, b.c0, b.a1, b.b1, b.c1)
        bvh = build([a, b])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertEqual(intersect(bvh, ray, 0.0).face_id, 0)

    def test_strictly_closer_beats_tie_rule(self):
        near = Triangle.stationary(
            (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 1.0, 1.0), face_id=99)
        far = Triangle.stationary(
            (-1.0, -1.0, 3.0), (1.0, -1.0, 3.0), (0.0, 1.0, 3.0), face_id=1)
        bvh = build([near, far])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect(bvh, ray, 0.0)
        self.assertEqual(res.face_id, 99)
        self.assertAlmostEqual(res.t, 1.0, places=12)


class TestRayWindowAndDirection(unittest.TestCase):
    def setUp(self):
        self.tri = Triangle.stationary(
            (-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0))
        self.bvh = build([self.tri])
        self.ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    def test_t_min_t_max_clipping(self):
        self.assertFalse(intersect(self.bvh, self.ray, 0.0, t_min=6.0).hit)
        self.assertFalse(intersect(self.bvh, self.ray, 0.0, t_max=4.0).hit)
        # 端点恰好包含
        res = intersect(self.bvh, self.ray, 0.0, t_min=5.0, t_max=5.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 5.0, places=12)
        res = intersect(self.bvh, self.ray, 0.0, t_max=math.inf)
        self.assertTrue(res.hit)

    def test_negative_t_min_allows_behind_origin(self):
        # 三角形在起点后方 z=-5，负方向射线，t_min 允许负值
        tri = Triangle.stationary(
            (-1.0, -1.0, -5.0), (1.0, -1.0, -5.0), (0.0, 1.0, -5.0))
        bvh = build([tri])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertFalse(intersect(bvh, ray, 0.0).hit)  # 默认 t_min=0
        res = intersect(bvh, ray, 0.0, t_min=-10.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, -5.0, places=10)
        _assert_ref_agreement(self, bvh, [tri], ray, 0.0, t_min=-10.0)

    def test_t_max_is_hard_despite_tie_band_window(self):
        # 单一面位于 t_max 之外 17 ulp：超出 16 ulp 同距带，但在 18 ulp
        # 剪枝放宽窗口内。t_max=5.0 时必须硬拒绝，不得报出 t>t_max 的命中。
        eps = 2.220446049250313e-16
        z = 5.0 + 17 * eps * 5.0
        self.assertGreater(z, 5.0)
        tri = Triangle.stationary(
            (-1.0, -1.0, z), (1.0, -1.0, z), (0.0, 1.0, z))
        bvh = build([tri])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect(bvh, ray, 0.0, t_max=5.0)
        self.assertFalse(res.hit)
        # 同一面在放宽后的 t_max 下正常命中
        res2 = intersect(bvh, ray, 0.0, t_max=6.0)
        self.assertTrue(res2.hit)
        self.assertGreater(res2.t, 5.0)
        _assert_ref_agreement(self, bvh, [tri], ray, 0.0, t_max=5.0)

    def test_zero_direction_components(self):
        # 方向只有一个非零分量（另两个分量严格为 0.0），三角形垂直于该轴
        cases = [
            ((0.0, 0.0, 1.0),
             ((-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0))),
            ((0.0, 1.0, 0.0),
             ((-1.0, 5.0, -1.0), (1.0, 5.0, -1.0), (0.0, 5.0, 1.0))),
            ((1.0, 0.0, 0.0),
             ((5.0, -1.0, -1.0), (5.0, 1.0, -1.0), (5.0, 0.0, 1.0))),
        ]
        for d, (a, b, c) in cases:
            bvh = build([Triangle.stationary(a, b, c)])
            res = intersect(bvh, Ray((0.0, 0.0, 0.0), d), 0.0)
            self.assertTrue(res.hit, f"方向 {d} 应命中")
            self.assertAlmostEqual(res.t, 5.0, places=10)
            _assert_ref_agreement(
                self, bvh, [Triangle.stationary(a, b, c)],
                Ray((0.0, 0.0, 0.0), d), 0.0)

    def test_negative_direction(self):
        tri = Triangle.stationary(
            (-1.0, -1.0, -5.0), (1.0, -1.0, -5.0), (0.0, 1.0, -5.0))
        bvh = build([tri])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        res = intersect(bvh, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 5.0, places=10)

    def test_double_sided(self):
        tri = Triangle.stationary(
            (-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0))
        bvh = build([tri])
        # 从背面射来
        ray = Ray((0.0, 0.0, 10.0), (0.0, 0.0, -1.0))
        res = intersect(bvh, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 5.0, places=10)

    def test_parallel_and_coplanar_ray_misses(self):
        ray = Ray((0.0, 0.0, 5.0), (1.0, 0.0, 0.0))  # 与三角形共面
        self.assertFalse(intersect(self.bvh, ray, 0.0).hit)
        ray2 = Ray((0.0, 0.0, 6.0), (1.0, 0.0, 0.0))  # 平行且面外
        self.assertFalse(intersect(self.bvh, ray2, 0.0).hit)


class TestCullingCounts(unittest.TestCase):
    def _two_clusters(self):
        tris = []
        for i in range(24):
            tris.append(Triangle.stationary(
                (i * 0.5, 0.0, 20.0), (i * 0.5 + 0.4, 0.0, 20.0),
                (i * 0.5 + 0.2, 1.0, 20.0)))
        for i in range(24):
            tris.append(Triangle.stationary(
                (i * 0.5, 0.0, -20.0), (i * 0.5 + 0.4, 0.0, -20.0),
                (i * 0.5 + 0.2, 1.0, -20.0)))
        return tris

    def test_fewer_triangles_tested_than_brute_force(self):
        tris = self._two_clusters()
        bvh = build(tris)
        # 射向 +z 簇第 13 个三角形的质心 (i*0.5+0.2, 1/3, 20)：
        # -z 簇整体应被剪枝，+z 簇内 x 方向也只测少数叶
        ray = Ray((6.2, 0.3333333333333333, 0.0), (0.0, 0.0, 1.0))
        res = intersect(bvh, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertLess(res.triangles_tested, len(tris),
                        "BVH 必须比逐面暴力参考少测面")
        self.assertLessEqual(res.triangles_tested, 32)
        ref = reference_intersect(tris, ray, 0.0)
        self.assertEqual(ref["triangles_tested"], len(tris))

    def test_proved_miss_has_zero_triangle_tests(self):
        tris = self._two_clusters()
        bvh = build(tris)
        ray = Ray((1000.0, 1000.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect(bvh, ray, 0.0)
        self.assertFalse(res.hit)
        self.assertEqual(res.triangles_tested, 0)
        self.assertGreaterEqual(res.nodes_visited, 1)  # 根节点一定访问

    def test_counts_are_monotone_and_present_on_miss(self):
        rng = random.Random(99)
        tris = [_rand_tri(rng) for _ in range(50)]
        bvh = build(tris)
        ray = _wild_ray(rng)
        res = intersect(bvh, ray, 0.5)
        self.assertGreaterEqual(res.nodes_visited, 1)
        self.assertGreaterEqual(res.triangles_tested, 0)
        self.assertLessEqual(res.triangles_tested, len(tris))


class TestStructure(unittest.TestCase):
    def test_leaf_capacity(self):
        rng = random.Random(3)
        for n in (1, 4, 5, 8, 33, 100, 300):
            tris = [_rand_tri(rng) for _ in range(n)]
            bvh = build(tris)

            def walk(node, depth=0):
                if node.faces is not None:
                    self.assertLessEqual(len(node.faces), mb.LEAF_SIZE)
                    self.assertIsNone(node.left)
                    return 1
                return 1 + walk(node.left, depth + 1) + walk(node.right, depth + 1)

            total = walk(bvh._root)
            self.assertGreaterEqual(total, 1)

    def test_endpoint_boxes_contain_endpoint_geometry(self):
        rng = random.Random(5)
        tris = [_rand_tri(rng) for _ in range(60)]
        bvh = build(tris)

        def check(node):
            if node.faces is not None:
                for idx in node.faces:
                    f = bvh._faces[idx]
                    for pi in range(6):
                        p = f[pi]
                        box = (node.lo0, node.hi0) if pi < 3 else (node.lo1, node.hi1)
                        for k in range(3):
                            self.assertGreaterEqual(p[k], box[0][k] - 1e-12)
                            self.assertLessEqual(p[k], box[1][k] + 1e-12)
            else:
                check(node.left)
                check(node.right)

        check(bvh._root)


# ---------------------------------------------------------------------------
# 顶点更新（拓扑不变、自底向上 refit）
# ---------------------------------------------------------------------------

def _leaf_faces(node):
    if node.faces is not None:
        return [node.faces]
    return _leaf_faces(node.left) + _leaf_faces(node.right)


def _assert_same_result(test, got, exp):
    """两次查询结果（hit/face_id/t/u/v/point）完全一致；计数不作要求。"""
    test.assertIs(got.hit, exp.hit)
    test.assertEqual(got.face_id, exp.face_id)
    if not got.hit:
        return
    # 同一三角形、同一射线、同一公式：t/u/v/point 应逐位一致
    test.assertEqual(got.t, exp.t)
    test.assertEqual(got.u, exp.u)
    test.assertEqual(got.v, exp.v)
    test.assertEqual(got.point, exp.point)


class TestVertexUpdate(unittest.TestCase):
    """update_vertices：拓扑不变的顶点位置更新 + 自底向上 refit。"""

    def _scene(self, rng, n, motion_scale=2.0):
        return [_rand_tri(rng, motion_scale) for _ in range(n)]

    def _check_against_rebuild_and_reference(self, bvh, tris, rng, n_rays=40):
        rebuilt = build(tris)
        for _ in range(n_rays):
            tau = rng.choice((0.0, 0.31, 0.5, 0.77, 1.0))
            if rng.random() < 0.7:
                ray = _aimed_ray(rng, tris, tau)
            else:
                ray = _wild_ray(rng)
            got = intersect(bvh, ray, tau)
            # 与全量重建一致（解析关系：refit 后查询 == 重建后查询）
            _assert_same_result(self, got, intersect(rebuilt, ray, tau))
            # 与独立暴力参考一致（独立期望值来源）
            _assert_ref_agreement(self, bvh, tris, ray, tau)

    def test_refit_matches_rebuild_and_reference(self):
        rng = random.Random(20260924)
        for n in (1, 4, 5, 40, 300):
            tris = self._scene(rng, n)
            bvh = build(tris)
            new_tris = self._scene(rng, n, motion_scale=6.0)  # 全新几何
            bvh.update_vertices(new_tris)
            self.assertEqual(bvh.triangle_count, n)
            self._check_against_rebuild_and_reference(bvh, new_tris, rng)

    def test_module_level_entry_and_iterable_input(self):
        rng = random.Random(31)
        tris = self._scene(rng, 9)
        bvh = build(tris)
        new_tris = self._scene(rng, 9)
        # 模块级入口 + 生成器（可迭代）输入
        mb.update_vertices(bvh, (t for t in new_tris))
        self._check_against_rebuild_and_reference(bvh, new_tris, rng, n_rays=20)

    def test_move_across_old_box_hand_computed(self):
        # 手算例子：三角形整体平移 +100x，远远跨出旧包围盒。
        # 新三角形 a=(99,-1,5) b=(101,-1,5) c=(100,1,5)；射线 (100,0,0)+t(0,0,1)
        # 交点 (100,0,5)：P-a=(1,1,0)=u(2,0,0)+v(1,2,0) → u=0.25, v=0.5, t=5。
        tri = Triangle.stationary((-1.0, -1.0, 5.0), (1.0, -1.0, 5.0),
                                  (0.0, 1.0, 5.0))
        bvh = build([tri])
        ray_old = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertTrue(intersect(bvh, ray_old, 0.0).hit)

        new = Triangle.stationary((99.0, -1.0, 5.0), (101.0, -1.0, 5.0),
                                  (100.0, 1.0, 5.0))
        bvh.update_vertices([new])
        # 旧位置不再命中（包围盒已随 refit 搬离）
        res_old = intersect(bvh, ray_old, 0.0)
        self.assertFalse(res_old.hit)
        self.assertEqual(res_old.triangles_tested, 0)  # 盒级即被排除
        # 新位置命中，手算期望值
        res = intersect(bvh, Ray((100.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 5.0, places=12)
        self.assertAlmostEqual(res.u, 0.25, places=12)
        self.assertAlmostEqual(res.v, 0.5, places=12)
        self.assertAlmostEqual(res.point[0], 100.0, places=10)
        self.assertAlmostEqual(res.point[2], 5.0, places=10)
        _assert_ref_agreement(self, bvh, [new],
                              Ray((100.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0.0)

    def test_update_motion_endpoints_changes_timeline(self):
        # 只改 t=1 端点：τ=0 结果不变，τ=1 跟随新端点（手算）
        tri = Triangle.stationary((-1.0, -1.0, 5.0), (1.0, -1.0, 5.0),
                                  (0.0, 1.0, 5.0))
        bvh = build([tri])
        moved = Triangle((-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0),
                         (9.0, -1.0, 5.0), (11.0, -1.0, 5.0), (10.0, 1.0, 5.0))
        bvh.update_vertices([moved])
        ray0 = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertTrue(intersect(bvh, ray0, 0.0).hit)   # τ=0 仍在原处
        self.assertFalse(intersect(bvh, ray0, 1.0).hit)  # τ=1 已移到 x+10
        res = intersect(bvh, Ray((10.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 1.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 5.0, places=12)
        self.assertAlmostEqual(res.u, 0.25, places=12)
        self.assertAlmostEqual(res.v, 0.5, places=12)
        # τ=0.5：线性插值到 x+5
        res = intersect(bvh, Ray((5.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0.5)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.point[0], 5.0, places=10)
        _assert_ref_agreement(self, bvh, [moved],
                              Ray((5.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0.5)

    def test_update_into_and_out_of_degenerate(self):
        # 正常 → 退化（共线零面积）：不再命中；退化 → 正常：恢复命中
        good = Triangle.stationary((-1.0, -1.0, 5.0), (1.0, -1.0, 5.0),
                                   (0.0, 1.0, 5.0))
        bvh = build([good])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertTrue(intersect(bvh, ray, 0.0).hit)

        degen = Triangle.stationary((-1.0, 0.0, 5.0), (0.0, 0.0, 5.0),
                                    (1.0, 0.0, 5.0))  # 三顶点共线
        bvh.update_vertices([degen])
        res = intersect(bvh, ray, 0.0)
        self.assertFalse(res.hit)
        self.assertEqual(res.triangles_tested, 1)  # 进了叶、被 det==0 拒绝
        _assert_ref_agreement(self, bvh, [degen], ray, 0.0)

        bvh.update_vertices([good])
        res = intersect(bvh, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 5.0, places=12)

    def test_degenerate_only_at_some_time(self):
        # t=0 正常、t=1 退化为共线：τ=0 命中、τ=1 不命中、中间时刻与参考一致
        tri = Triangle((-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0),
                       (-1.0, 0.0, 5.0), (1.0, 0.0, 5.0), (0.0, 0.0, 5.0))
        bvh = build([tri])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertTrue(intersect(bvh, ray, 0.0).hit)
        self.assertFalse(intersect(bvh, ray, 1.0).hit)
        for tau in (0.25, 0.5, 0.75):
            _assert_ref_agreement(self, bvh, [tri], ray, tau)

    def test_two_consecutive_updates(self):
        rng = random.Random(20260925)
        n = 60
        tris0 = self._scene(rng, n)
        bvh = build(tris0)
        # 第一次更新：含跨旧包围盒的大位移
        tris1 = self._scene(rng, n, motion_scale=8.0)
        bvh.update_vertices(tris1)
        self._check_against_rebuild_and_reference(bvh, tris1, rng, n_rays=30)
        # 第二次更新：在第一次的树上继续（含退化面混入）
        tris2 = self._scene(rng, n)
        tris2[7] = Triangle.stationary((0.0, 0.0, 3.0), (0.0, 0.0, 3.0),
                                       (0.0, 0.0, 3.0))  # 点三角形
        bvh.update_vertices(tris2)
        self.assertEqual(bvh.triangle_count, n)
        self._check_against_rebuild_and_reference(bvh, tris2, rng, n_rays=30)

    def test_topology_and_leaf_membership_unchanged(self):
        rng = random.Random(13)
        tris = self._scene(rng, 50)
        bvh = build(tris)
        before = _leaf_faces(bvh._root)
        new_tris = self._scene(rng, 50, motion_scale=5.0)
        bvh.update_vertices(new_tris)
        # 树结构复用：叶成员（面序号集合与顺序）完全一致
        self.assertEqual(before, _leaf_faces(bvh._root))
        # refit 后的两端盒包含新几何（逐节点、逐端点顶点）
        def check(node):
            if node.faces is not None:
                for idx in node.faces:
                    f = bvh._faces[idx]
                    for pi in range(6):
                        p = f[pi]
                        box = (node.lo0, node.hi0) if pi < 3 else (node.lo1, node.hi1)
                        for k in range(3):
                            self.assertGreaterEqual(p[k], box[0][k] - 1e-12)
                            self.assertLessEqual(p[k], box[1][k] + 1e-12)
            else:
                # 内部节点盒 == 子节点盒并集（精确）
                for k in range(3):
                    self.assertEqual(node.lo0[k], min(node.left.lo0[k], node.right.lo0[k]))
                    self.assertEqual(node.hi0[k], max(node.left.hi0[k], node.right.hi0[k]))
                    self.assertEqual(node.lo1[k], min(node.left.lo1[k], node.right.lo1[k]))
                    self.assertEqual(node.hi1[k], max(node.left.hi1[k], node.right.hi1[k]))
                check(node.left)
                check(node.right)
        check(bvh._root)

    def test_custom_face_ids_preserved(self):
        rng = random.Random(77)
        tris = self._scene(rng, 20)
        ids = list(range(500, 520))
        rng.shuffle(ids)
        tris = [Triangle(t.a0, t.b0, t.c0, t.a1, t.b1, t.c1, face_id=fid)
                for t, fid in zip(tris, ids)]
        bvh = build(tris)
        # 更新时显式给出相同 face_id：合法
        new_tris = self._scene(rng, 20)
        new_tris = [Triangle(t.a0, t.b0, t.c0, t.a1, t.b1, t.c1, face_id=fid)
                    for t, fid in zip(new_tris, ids)]
        bvh.update_vertices(new_tris)
        self._check_against_rebuild_and_reference(bvh, new_tris, rng, n_rays=25)

    def test_empty_bvh_update(self):
        bvh = build([])
        bvh.update_vertices([])  # 空对空：合法无操作
        res = intersect(bvh, Ray((0, 0, 0), (0, 0, 1)), 0.0)
        self.assertFalse(res.hit)
        with self.assertRaises(ValueError):
            bvh.update_vertices([Triangle.stationary((0, 0, 0), (1, 0, 0),
                                                     (0, 1, 0))])


class TestVertexUpdateValidation(unittest.TestCase):
    """更新输入非法：明确报错且旧树完全保持（全有或全无）。"""

    def _base(self):
        # 三个静止三角形，分别位于 z=5 / z=10 / z=15，便于逐面定位查询
        tris = [
            Triangle.stationary((-1.0, -1.0, 5.0), (1.0, -1.0, 5.0),
                                (0.0, 1.0, 5.0)),
            Triangle.stationary((-1.0, -1.0, 10.0), (1.0, -1.0, 10.0),
                                (0.0, 1.0, 10.0)),
            Triangle.stationary((-1.0, -1.0, 15.0), (1.0, -1.0, 15.0),
                                (0.0, 1.0, 15.0)),
        ]
        return tris, build(tris)

    def _probe(self, bvh):
        # 手算：射线 (0,0,0)+t(0,0,1) 依次穿过三个面，最近命中 z=5，t=5
        return intersect(bvh, Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0.0)

    def test_reject_count_mismatch(self):
        tris, bvh = self._base()
        with self.assertRaises(ValueError):
            bvh.update_vertices(tris[:2])
        with self.assertRaises(ValueError):
            bvh.update_vertices(tris + tris[:1])
        res = self._probe(bvh)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 5.0, places=12)

    def test_reject_nan_infinity_and_keep_old_tree(self):
        tris, bvh = self._base()
        before = self._probe(bvh)
        # 非法项放在最后：前两个合法也不得生效（全有或全无）
        moved = [
            Triangle.stationary((99.0, -1.0, 5.0), (101.0, -1.0, 5.0),
                                (100.0, 1.0, 5.0)),
            Triangle.stationary((99.0, -1.0, 10.0), (101.0, -1.0, 10.0),
                                (100.0, 1.0, 10.0)),
            Triangle.stationary((0.0, 0.0, 0.0), (math.nan, 0.0, 0.0),
                                (0.0, 1.0, 0.0)),
        ]
        with self.assertRaises(ValueError):
            bvh.update_vertices(moved)
        after = self._probe(bvh)
        _assert_same_result(self, after, before)
        self.assertAlmostEqual(after.t, 5.0, places=12)  # 未搬去 x+100
        for bad in (math.inf, -math.inf):
            bad_tris = list(tris)
            bad_tris[1] = Triangle.stationary(
                (0.0, 0.0, bad), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
            with self.assertRaises(ValueError):
                bvh.update_vertices(bad_tris)
        _assert_same_result(self, self._probe(bvh), before)

    def test_reject_wrong_types(self):
        tris, bvh = self._base()
        before = self._probe(bvh)
        with self.assertRaises(TypeError):
            bvh.update_vertices("not a list")
        with self.assertRaises(TypeError):
            bvh.update_vertices([tris[0], (0, 1, 2), tris[2]])
        with self.assertRaises(TypeError):
            bvh.update_vertices([tris[0], tris[1],
                                 Triangle((0, 0), (1, 0, 0), (0, 1, 0),
                                          (0, 0, 0), (1, 0, 0), (0, 1, 0))])
        with self.assertRaises(TypeError):
            bvh.update_vertices([Triangle((True, 0.0, 0.0), (1, 0, 0), (0, 1, 0),
                                          (0, 0, 0), (1, 0, 0), (0, 1, 0))] * 3)
        with self.assertRaises(TypeError):
            mb.update_vertices("bvh", tris)
        _assert_same_result(self, self._probe(bvh), before)

    def test_reject_face_id_change(self):
        tris = [Triangle.stationary((-1.0, -1.0, 5.0), (1.0, -1.0, 5.0),
                                    (0.0, 1.0, 5.0), face_id=7),
                Triangle.stationary((-1.0, -1.0, 10.0), (1.0, -1.0, 10.0),
                                    (0.0, 1.0, 10.0))]  # 默认 id=1
        bvh = build(tris)
        before = self._probe(bvh)
        # 显式 face_id 与现有不一致 → ValueError
        bad = Triangle.stationary((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                  (0.0, 1.0, 0.0), face_id=8)
        with self.assertRaises(ValueError):
            bvh.update_vertices([bad, tris[1]])
        # 负 face_id → ValueError；非 int → TypeError
        worse = Triangle.stationary((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                    (0.0, 1.0, 0.0), face_id=-2)
        with self.assertRaises(ValueError):
            bvh.update_vertices([worse, tris[1]])
        with self.assertRaises(TypeError):
            bvh.update_vertices([Triangle.stationary(
                (0, 0, 0), (1, 0, 0), (0, 1, 0), face_id=1.5), tris[1]])
        _assert_same_result(self, self._probe(bvh), before)
        # face_id=None 表示沿用：合法
        bvh.update_vertices([Triangle.stationary(
            (-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0)), tris[1]])
        res = self._probe(bvh)
        self.assertTrue(res.hit)
        self.assertEqual(res.face_id, 7)  # 标识保持不变

    def test_error_is_locatable(self):
        tris, bvh = self._base()
        bad = list(tris)
        bad[2] = Triangle.stationary((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                     (0.0, math.nan, 0.0))
        try:
            bvh.update_vertices(bad)
        except ValueError as exc:
            self.assertIn("triangles[2]", str(exc))
        else:
            self.fail("应抛出 ValueError")


class TestEmpty(unittest.TestCase):
    def test_empty_bvh(self):
        bvh = build([])
        self.assertEqual(bvh.triangle_count, 0)
        res = intersect(bvh, Ray((0, 0, 0), (0, 0, 1)), 0.0)
        self.assertFalse(res.hit)
        self.assertEqual(res.nodes_visited, 0)
        self.assertEqual(res.triangles_tested, 0)
        self.assertIsNone(res.face_id)
        self.assertTrue(math.isnan(res.t))


class TestValidation(unittest.TestCase):
    def _ok(self):
        return Triangle.stationary(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    # --- 数值非法：NaN / Infinity --------------------------------------

    def test_reject_nan_infinity_in_vertices(self):
        T = Triangle
        bad_vals = (math.nan, math.inf, -math.inf)
        fields = ["a0", "b0", "c0", "a1", "b1", "c1"]
        for fld in fields:
            for bad in bad_vals:
                kw = dict(a0=(0, 0, 0), b0=(1, 0, 0), c0=(0, 1, 0),
                          a1=(0, 0, 0), b1=(1, 0, 0), c1=(0, 1, 0))
                p = list(kw[fld])
                p[0] = bad
                kw[fld] = tuple(p)
                with self.assertRaises(ValueError):
                    build([T(**kw)])

    def test_reject_nan_infinity_in_ray_and_params(self):
        bvh = build([self._ok()])
        ray = Ray((math.nan, 0, 0), (0, 0, 1))
        with self.assertRaises(ValueError):
            intersect(bvh, ray, 0.0)
        with self.assertRaises(ValueError):
            intersect(bvh, Ray((0, 0, 0), (math.inf, 0, 1)), 0.0)
        with self.assertRaises(ValueError):
            intersect(bvh, Ray((0, 0, 0), (0, 0, 1)), math.nan)
        with self.assertRaises(ValueError):
            intersect(bvh, Ray((0, 0, 0), (0, 0, 1)), 0.0, t_min=math.inf)
        with self.assertRaises(ValueError):
            intersect(bvh, Ray((0, 0, 0), (0, 0, 1)), 0.0, t_max=math.nan)
        with self.assertRaises(ValueError):
            intersect(bvh, Ray((0, 0, 0), (0, 0, 1)), 0.0, t_max=-math.inf)

    # --- bool 拒绝 ------------------------------------------------------

    def test_reject_bool_for_numbers_and_ints(self):
        with self.assertRaises(TypeError):
            build([Triangle((True, 0.0, 0.0), (1, 0, 0), (0, 1, 0),
                            (0, 0, 0), (1, 0, 0), (0, 1, 0))])
        bvh = build([self._ok()])
        with self.assertRaises(TypeError):
            intersect(bvh, Ray((0, 0, 0), (0, 0, 1)), True)
        with self.assertRaises(TypeError):
            build([Triangle.stationary((0, 0, 0), (1, 0, 0), (0, 1, 0),
                                       face_id=True)])

    # --- 类型/形状 ------------------------------------------------------

    def test_reject_wrong_types(self):
        with self.assertRaises(TypeError):
            build("not a list")
        with self.assertRaises(TypeError):
            build([(0, 1, 2)])
        with self.assertRaises(TypeError):
            intersect("bvh", Ray((0, 0, 0), (0, 0, 1)), 0.0)
        with self.assertRaises(TypeError):
            intersect(build([self._ok()]), ((0, 0, 0), (0, 0, 1)), 0.0)
        with self.assertRaises(TypeError):
            build([Triangle((0, 0), (1, 0, 0), (0, 1, 0),
                            (0, 0, 0), (1, 0, 0), (0, 1, 0))])
        with self.assertRaises(TypeError):
            build([Triangle("x", (1, 0, 0), (0, 1, 0),
                            (0, 0, 0), (1, 0, 0), (0, 1, 0))])

    # --- 域约束 ---------------------------------------------------------

    def test_reject_time_out_of_range(self):
        bvh = build([self._ok()])
        ray = Ray((0, 0, 0), (0, 0, 1))
        for bad in (-0.0001, 1.0001, -1.0, 2.0):
            with self.assertRaises(ValueError):
                intersect(bvh, ray, bad)

    def test_reject_zero_direction(self):
        bvh = build([self._ok()])
        with self.assertRaises(ValueError):
            intersect(bvh, Ray((0, 0, 0), (0, 0, 0)), 0.0)

    def test_reject_t_min_greater_than_t_max(self):
        bvh = build([self._ok()])
        with self.assertRaises(ValueError):
            intersect(bvh, Ray((0, 0, 0), (0, 0, 1)), 0.0, t_min=5, t_max=4)

    def test_triangle_limit(self):
        rng = random.Random(1)
        tris300 = [_rand_tri(rng) for _ in range(300)]
        bvh = build(tris300)  # 恰好上限，合法
        self.assertEqual(bvh.triangle_count, 300)
        tris301 = tris300 + [self._ok()]
        with self.assertRaises(ValueError):
            build(tris301)

    def test_reject_duplicate_and_negative_face_id(self):
        a = self._ok()
        b = Triangle.stationary((2, 0, 0), (3, 0, 0), (2, 1, 0), face_id=7)
        a = Triangle(a.a0, a.b0, a.c0, a.a1, a.b1, a.c1, face_id=7)
        with self.assertRaises(ValueError):
            build([a, b])
        c = Triangle.stationary((2, 0, 0), (3, 0, 0), (2, 1, 0), face_id=-1)
        with self.assertRaises(ValueError):
            build([c])

    def test_reject_non_int_face_id(self):
        for bad in (1.0, "7", None):
            # None 是合法默认值，跳过
            if bad is None:
                continue
            with self.assertRaises(TypeError):
                build([Triangle.stationary(
                    (0, 0, 0), (1, 0, 0), (0, 1, 0), face_id=bad)])

    def test_error_is_locatable(self):
        # 错误信息应指出出错字段/位置
        try:
            build([self._ok(),
                   Triangle.stationary((0, 0, 0), (math.nan, 0, 0), (0, 1, 0))])
        except ValueError as exc:
            self.assertIn("triangles[1]", str(exc))
        else:
            self.fail("应抛出 ValueError")

    def test_failed_build_leaves_nothing_behind(self):
        # 构建是纯构造：失败调用前后，既有 BVH 行为不变
        good = build([self._ok()])
        before = intersect(good, Ray((0.2, 0.3, -1), (0, 0, 1)), 0.0)
        self.assertTrue(before.hit)
        with self.assertRaises(ValueError):
            build([self._ok()] * 12 + [
                Triangle.stationary((0, 0, 0), (1, 0, 0), (math.nan, 1, 0))])
        after = intersect(good, Ray((0.2, 0.3, -1), (0, 0, 1)), 0.0)
        self.assertEqual(after.face_id, before.face_id)
        self.assertEqual(after.t, before.t)

    def test_bvh_not_directly_constructible(self):
        with self.assertRaises(TypeError):
            mb.BVH()


if __name__ == "__main__":
    unittest.main(verbosity=2)
