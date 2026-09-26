"""closest_point 必要测试。

期望值来源：tests/reference.py 中独立书写的逐三角形投影参考
（reference_closest_point，不调用 motion_bvh 的任何算法）。
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
from motion_bvh import Triangle, Ray, build, closest_point
from reference import reference_closest_point


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


class TestClosestPointBasic(unittest.TestCase):
    def setUp(self):
        # z=0 平面上单个静止三角形 (0,0,0)-(2,0,0)-(0,2,0)
        self.tri = Triangle.stationary((0, 0, 0), (2, 0, 0), (0, 2, 0))
        self.bvh = build([self.tri])

    def test_projection_inside_face(self):
        r = closest_point(self.bvh, (0.5, 0.5, 3.0), 0.0)
        self.assertTrue(r.hit)
        self.assertEqual(r.face_id, 0)
        self.assertAlmostEqual(r.u, 0.25, places=12)
        self.assertAlmostEqual(r.v, 0.25, places=12)
        self.assertAlmostEqual(r.dist_sq, 9.0, places=12)
        self.assertAlmostEqual(r.point[2], 0.0, places=12)
        self.assertAlmostEqual(r.point[0], 0.5, places=12)
        self.assertAlmostEqual(r.point[1], 0.5, places=12)

    def test_projection_on_edge(self):
        # 斜边 x+y=2 外法向一侧的点：最近点落在边 bc 上
        r = closest_point(self.bvh, (2.0, 2.0, 0.0), 0.0)
        self.assertTrue(r.hit)
        self.assertAlmostEqual(r.u, 0.5, places=12)
        self.assertAlmostEqual(r.v, 0.5, places=12)
        self.assertAlmostEqual(r.dist_sq, 2.0, places=12)
        self.assertAlmostEqual(r.point[0], 1.0, places=12)
        self.assertAlmostEqual(r.point[1], 1.0, places=12)

    def test_projection_on_vertex(self):
        r = closest_point(self.bvh, (-1.0, -1.0, 1.0), 0.0)
        self.assertTrue(r.hit)
        self.assertAlmostEqual(r.u, 0.0, places=12)
        self.assertAlmostEqual(r.v, 0.0, places=12)
        self.assertAlmostEqual(r.dist_sq, 3.0, places=12)
        self.assertEqual(r.point, (0.0, 0.0, 0.0))

    def test_point_on_surface(self):
        r = closest_point(self.bvh, (0.5, 0.5, 0.0), 0.0)
        self.assertTrue(r.hit)
        self.assertAlmostEqual(r.dist_sq, 0.0, places=24)

    def test_motion_endpoints(self):
        # 三角形整体沿 +x 平移 10：τ=0 与 τ=1 的最近点应精确跟随端点
        tri = Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0),
                       (10, 0, 0), (11, 0, 0), (10, 1, 0))
        bvh = build([tri])
        r0 = closest_point(bvh, (0.25, 0.25, 5.0), 0.0)
        r1 = closest_point(bvh, (10.25, 0.25, 5.0), 1.0)
        self.assertAlmostEqual(r0.point[0], 0.25, places=12)
        self.assertAlmostEqual(r1.point[0], 10.25, places=12)
        self.assertAlmostEqual(r0.dist_sq, 25.0, places=12)
        self.assertAlmostEqual(r1.dist_sq, 25.0, places=12)
        # 中间时刻：最近点随运动插值
        rm = closest_point(bvh, (5.25, 0.25, 5.0), 0.5)
        self.assertAlmostEqual(rm.point[0], 5.25, places=12)
        self.assertAlmostEqual(rm.dist_sq, 25.0, places=12)


class TestDegenerate(unittest.TestCase):
    def test_segment_triangle(self):
        # 三顶点共线：退化为线段 (0,0,0)-(2,0,0)
        tri = Triangle.stationary((0, 0, 0), (2, 0, 0), (1, 0, 0))
        bvh = build([tri])
        r = closest_point(bvh, (0.5, 1.0, 0.0), 0.0)
        self.assertTrue(r.hit)
        self.assertAlmostEqual(r.dist_sq, 1.0, places=12)
        self.assertAlmostEqual(r.point[0], 0.5, places=12)
        self.assertAlmostEqual(r.point[1], 0.0, places=12)
        # 线段端点外：最近点为端点
        r2 = closest_point(bvh, (-3.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(r2.dist_sq, 9.0, places=12)
        self.assertEqual(r2.point, (0.0, 0.0, 0.0))

    def test_point_triangle(self):
        # 三顶点共点：退化为点
        tri = Triangle.stationary((1, 2, 3), (1, 2, 3), (1, 2, 3))
        bvh = build([tri])
        r = closest_point(bvh, (4, 6, 3), 0.0)
        self.assertTrue(r.hit)
        self.assertEqual(r.face_id, 0)
        self.assertAlmostEqual(r.u, 0.0, places=12)
        self.assertAlmostEqual(r.v, 0.0, places=12)
        self.assertAlmostEqual(r.dist_sq, 25.0, places=12)
        self.assertEqual(r.point, (1.0, 2.0, 3.0))

    def test_degenerate_motion_endpoint(self):
        # τ=1 时退化为点的运动三角形
        tri = Triangle((0, 0, 0), (2, 0, 0), (0, 2, 0),
                       (5, 5, 5), (5, 5, 5), (5, 5, 5))
        bvh = build([tri])
        r = closest_point(bvh, (5, 5, 8), 1.0)
        self.assertTrue(r.hit)
        self.assertAlmostEqual(r.dist_sq, 9.0, places=12)
        self.assertEqual(r.point, (5.0, 5.0, 5.0))


class TestTieAndEmpty(unittest.TestCase):
    def test_tie_by_face_id(self):
        # 两个完全重合的三角形，等距时取 face_id 较小者
        t1 = Triangle.stationary((0, 0, 0), (1, 0, 0), (0, 1, 0), face_id=7)
        t2 = Triangle.stationary((0, 0, 0), (1, 0, 0), (0, 1, 0), face_id=3)
        bvh = build([t1, t2])
        r = closest_point(bvh, (0.25, 0.25, 1.0), 0.0)
        self.assertTrue(r.hit)
        self.assertEqual(r.face_id, 3)

    def test_empty_mesh(self):
        bvh = build([])
        r = closest_point(bvh, (0, 0, 0), 0.5)
        self.assertFalse(r.hit)
        self.assertIsNone(r.face_id)
        self.assertIsNone(r.point)
        self.assertTrue(math.isnan(r.dist_sq))
        self.assertEqual(r.nodes_visited, 0)
        self.assertEqual(r.triangles_tested, 0)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.bvh = build([Triangle.stationary((0, 0, 0), (1, 0, 0), (0, 1, 0))])

    def test_invalid_time_rejected(self):
        for bad in (-0.1, 1.1, math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError, msg=f"time={bad}"):
                closest_point(self.bvh, (0, 0, 0), bad)
        with self.assertRaises(TypeError):
            closest_point(self.bvh, (0, 0, 0), True)

    def test_invalid_point_rejected(self):
        with self.assertRaises(TypeError):
            closest_point(self.bvh, (0, 0), 0.0)
        with self.assertRaises(TypeError):
            closest_point(self.bvh, "abc", 0.0)
        with self.assertRaises(ValueError):
            closest_point(self.bvh, (0, math.nan, 0), 0.0)
        with self.assertRaises(ValueError):
            closest_point(self.bvh, (0, 0, math.inf), 0.0)
        with self.assertRaises(TypeError):
            closest_point(self.bvh, (0, True, 0), 0.0)

    def test_invalid_bvh_rejected(self):
        with self.assertRaises(TypeError):
            closest_point("not a bvh", (0, 0, 0), 0.0)

    def test_query_is_readonly(self):
        # 查询不改树：重复查询结果一致，且之后 intersect 仍正常
        q = (0.25, 0.25, 2.0)
        r1 = closest_point(self.bvh, q, 0.5)
        r2 = closest_point(self.bvh, q, 0.5)
        self.assertEqual(r1, r2)
        hit = self.bvh.intersect(Ray((0.25, 0.25, 2.0), (0, 0, -1)), 0.5)
        self.assertTrue(hit.hit)
        r3 = closest_point(self.bvh, q, 0.5)
        self.assertEqual(r1, r3)


class TestPruning(unittest.TestCase):
    def test_scattered_grid_prunes(self):
        # 分散网格：10x10 个彼此远离的三角形，查询点贴近其中一个角
        tris = []
        for i in range(10):
            for j in range(10):
                x, y = i * 10.0, j * 10.0
                tris.append(Triangle.stationary(
                    (x, y, 0), (x + 1, y, 0), (x, y + 1, 0)))
        bvh = build(tris)
        total_nodes = 0

        def count(node):
            nonlocal total_nodes
            if node is None:
                return
            total_nodes += 1
            count(node.left)
            count(node.right)
        count(bvh._root)

        r = closest_point(bvh, (0.2, 0.2, 1.0), 0.0)
        self.assertTrue(r.hit)
        self.assertEqual(r.face_id, 0)
        # 分支限界必须实际剪掉节点与面（不是无条件全测）
        self.assertLess(r.triangles_tested, len(tris))
        self.assertLess(r.nodes_visited, total_nodes)
        # 计数被如实报告
        self.assertGreater(r.nodes_visited, 0)
        self.assertGreater(r.triangles_tested, 0)

    def test_far_query_still_correct(self):
        # 查询点远离整个网格：结果仍须与暴力参考一致
        tris = []
        for i in range(10):
            for j in range(10):
                x, y = i * 10.0, j * 10.0
                tris.append(Triangle.stationary(
                    (x, y, 0), (x + 1, y, 0), (x, y + 1, 0)))
        bvh = build(tris)
        r = closest_point(bvh, (1000.0, -500.0, 30.0), 0.0)
        ref = reference_closest_point(tris, (1000.0, -500.0, 30.0), 0.0)
        self.assertEqual(r.face_id, ref["face_id"])
        self.assertAlmostEqual(r.dist_sq, ref["dist_sq"], delta=1e-6)


class TestDifferential(unittest.TestCase):
    def test_random_against_reference(self):
        rng = random.Random(20260926)
        for trial in range(60):
            n = rng.randint(1, 40)
            tris = [_rand_tri(rng) for _ in range(n)]
            # 掺入退化面（线段/点）
            if n >= 3 and trial % 3 == 0:
                p = (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-4, 4))
                q = (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-4, 4))
                m = tuple((p[k] + q[k]) / 2 for k in range(3))
                tris[1] = Triangle.stationary(p, q, m)      # 共线
                tris[2] = Triangle.stationary(p, p, p)      # 共点
            bvh = build(tris)
            tau = rng.choice([0.0, 1.0, rng.random()])
            pt = (rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-6, 6))
            got = closest_point(bvh, pt, tau)
            exp = reference_closest_point(tris, pt, tau)
            self.assertTrue(got.hit)
            self.assertEqual(got.face_id, exp["face_id"],
                             msg=f"trial={trial} tau={tau} pt={pt}")
            self.assertAlmostEqual(
                got.dist_sq, exp["dist_sq"],
                delta=1e-9 * max(1.0, abs(exp["dist_sq"])),
                msg=f"trial={trial}")
            self.assertAlmostEqual(got.u, exp["u"], delta=1e-9,
                                   msg=f"trial={trial}")
            self.assertAlmostEqual(got.v, exp["v"], delta=1e-9,
                                   msg=f"trial={trial}")
            for k in range(3):
                self.assertAlmostEqual(got.point[k], exp["point"][k],
                                       delta=1e-8, msg=f"trial={trial}")


if __name__ == "__main__":
    unittest.main()
