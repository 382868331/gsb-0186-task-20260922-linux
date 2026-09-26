"""closest_point 增量测试。

期望值来源：tests/reference.py 中独立书写的逐三角形投影暴力参考
（reference_closest_point，不调用 motion_bvh 的任何算法函数）。
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
from motion_bvh import Triangle, build, closest_point
from reference import reference_closest_point


def _rand_tri(rng, motion_scale=1.5):
    def pt():
        return (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-4, 4))

    a0, b0, c0 = pt(), pt(), pt()
    mv = [(rng.uniform(-motion_scale, motion_scale),
           rng.uniform(-motion_scale, motion_scale),
           rng.uniform(-motion_scale, motion_scale)) for _ in range(3)]
    return Triangle(
        a0, b0, c0,
        tuple(a0[k] + mv[0][k] for k in range(3)),
        tuple(b0[k] + mv[1][k] for k in range(3)),
        tuple(c0[k] + mv[2][k] for k in range(3)),
    )


def _scattered_grid(n_side=6, spacing=8.0):
    """分散网格：三角形彼此远离，便于观察剪枝。返回 Triangle 列表。"""
    tris = []
    fid = 0
    for i in range(n_side):
        for j in range(n_side):
            cx = i * spacing
            cy = j * spacing
            cz = ((i * 7 + j * 13) % 5) * spacing * 0.5
            s = 0.8
            a0 = (cx - s, cy - s, cz)
            b0 = (cx + s, cy - s, cz)
            c0 = (cx, cy + s, cz)
            a1 = (a0[0] + 0.3, a0[1], a0[2] + 0.2)
            b1 = (b0[0], b0[1] + 0.3, b0[2] - 0.1)
            c1 = (c0[0] - 0.2, c0[1], c0[2] + 0.1)
            tris.append(Triangle(a0, b0, c0, a1, b1, c1, face_id=fid))
            fid += 1
    return tris


class TestClosestPointGeometry(unittest.TestCase):
    """投影落点分类：面内 / 边上 / 顶点 / 运动端点。"""

    def setUp(self):
        self.tri = Triangle.stationary(
            (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0))
        self.bvh = build([self.tri])

    def test_projection_inside_face(self):
        hit = closest_point(self.bvh, (1.0, 1.0, 3.0), 0.5)
        self.assertTrue(hit.hit)
        self.assertEqual(hit.face_id, 0)
        self.assertAlmostEqual(hit.dist_sq, 9.0)
        self.assertAlmostEqual(hit.u, 0.25)
        self.assertAlmostEqual(hit.v, 0.25)
        self.assertAlmostEqual(hit.point[0], 1.0)
        self.assertAlmostEqual(hit.point[1], 1.0)
        self.assertAlmostEqual(hit.point[2], 0.0)

    def test_projection_on_edge(self):
        # 最近点落在斜边 BC 上：BC 为 x+y=4，最近点是 (2,2,0)
        hit = closest_point(self.bvh, (3.0, 3.0, 1.0), 0.5)
        self.assertTrue(hit.hit)
        self.assertAlmostEqual(hit.point[0], 2.0)
        self.assertAlmostEqual(hit.point[1], 2.0)
        self.assertAlmostEqual(hit.point[2], 0.0)
        self.assertAlmostEqual(hit.dist_sq, 3.0)
        self.assertAlmostEqual(hit.u + hit.v, 1.0)

    def test_projection_on_vertex(self):
        hit = closest_point(self.bvh, (-3.0, -4.0, 0.0), 0.5)
        self.assertTrue(hit.hit)
        self.assertEqual(hit.point, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(hit.dist_sq, 25.0)
        self.assertAlmostEqual(hit.u, 0.0)
        self.assertAlmostEqual(hit.v, 0.0)

    def test_motion_endpoints(self):
        # 三角形沿 z 从 0 运动到 10
        tri = Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0),
                       (0, 0, 10), (1, 0, 10), (0, 1, 10))
        bvh = build([tri])
        h0 = closest_point(bvh, (0.25, 0.25, -5.0), 0.0)
        self.assertAlmostEqual(h0.point[2], 0.0)
        self.assertAlmostEqual(h0.dist_sq, 25.0)
        h1 = closest_point(bvh, (0.25, 0.25, 99.0), 1.0)
        self.assertAlmostEqual(h1.point[2], 10.0)
        self.assertAlmostEqual(h1.dist_sq, 89.0 * 89.0)
        hm = closest_point(bvh, (0.25, 0.25, 5.0), 0.5)
        self.assertAlmostEqual(hm.dist_sq, 0.0)

    def test_point_exactly_on_surface(self):
        hit = closest_point(self.bvh, (1.0, 1.0, 0.0), 0.0)
        self.assertTrue(hit.hit)
        self.assertAlmostEqual(hit.dist_sq, 0.0)


class TestDegenerateTriangles(unittest.TestCase):
    """退化为线段或点的三角形也要给出结果。"""

    def test_segment_triangle(self):
        # 三顶点共线：退化为线段 (0,0,0)-(4,0,0)
        tri = Triangle.stationary((0, 0, 0), (4, 0, 0), (2, 0, 0))
        bvh = build([tri])
        hit = closest_point(bvh, (1.0, 3.0, 4.0), 0.5)
        self.assertTrue(hit.hit)
        self.assertAlmostEqual(hit.point[0], 1.0)
        self.assertAlmostEqual(hit.point[1], 0.0)
        self.assertAlmostEqual(hit.dist_sq, 25.0)

    def test_point_triangle(self):
        # 三顶点重合：退化为点
        tri = Triangle.stationary((1, 2, 3), (1, 2, 3), (1, 2, 3))
        bvh = build([tri])
        hit = closest_point(bvh, (4, 6, 3), 0.5)
        self.assertTrue(hit.hit)
        self.assertEqual(hit.point, (1.0, 2.0, 3.0))
        self.assertAlmostEqual(hit.dist_sq, 25.0)

    def test_degenerate_matches_reference(self):
        rng = random.Random(7)
        tris = []
        for i in range(6):
            x = rng.uniform(-3, 3)
            y = rng.uniform(-3, 3)
            z = rng.uniform(-3, 3)
            if i % 2 == 0:  # 线段
                d = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
                tris.append(Triangle.stationary(
                    (x, y, z),
                    (x + d[0], y + d[1], z + d[2]),
                    (x + 0.5 * d[0], y + 0.5 * d[1], z + 0.5 * d[2]),
                    face_id=i))
            else:  # 点
                tris.append(Triangle.stationary(
                    (x, y, z), (x, y, z), (x, y, z), face_id=i))
        bvh = build(tris)
        for _ in range(20):
            p = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5))
            tau = rng.choice([0.0, 0.25, 0.5, 1.0])
            exp = reference_closest_point(tris, p, tau)
            got = closest_point(bvh, p, tau)
            self.assertTrue(got.hit)
            self.assertEqual(got.face_id, exp["face_id"], (p, tau))
            self.assertAlmostEqual(got.dist_sq, exp["dist_sq"], places=9)


class TestTieAndContract(unittest.TestCase):
    def test_tie_breaks_by_face_id(self):
        # 两个关于查询点对称、完全等距的三角形，face_id 小者胜
        t_big = Triangle.stationary((0, 0, 5), (2, 0, 5), (0, 2, 5), face_id=9)
        t_small = Triangle.stationary((0, 0, -5), (0, 2, -5), (2, 0, -5),
                                      face_id=3)
        bvh = build([t_big, t_small])
        hit = closest_point(bvh, (0.5, 0.5, 0.0), 0.5)
        self.assertTrue(hit.hit)
        self.assertEqual(hit.face_id, 3)
        self.assertAlmostEqual(hit.dist_sq, 25.0)

    def test_empty_mesh_no_hit(self):
        bvh = build([])
        hit = closest_point(bvh, (0, 0, 0), 0.5)
        self.assertFalse(hit.hit)
        self.assertIsNone(hit.face_id)
        self.assertIsNone(hit.point)
        self.assertTrue(math.isnan(hit.dist_sq))
        self.assertEqual(hit.nodes_visited, 0)
        self.assertEqual(hit.triangles_tested, 0)

    def test_invalid_time_rejected(self):
        bvh = build([Triangle.stationary((0, 0, 0), (1, 0, 0), (0, 1, 0))])
        for bad in (-0.1, 1.1, math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError, msg=bad):
                closest_point(bvh, (0, 0, 0), bad)
        with self.assertRaises(TypeError):
            closest_point(bvh, (0, 0, 0), True)

    def test_invalid_point_rejected(self):
        bvh = build([Triangle.stationary((0, 0, 0), (1, 0, 0), (0, 1, 0))])
        with self.assertRaises(TypeError):
            closest_point(bvh, (0, 0), 0.5)
        with self.assertRaises(ValueError):
            closest_point(bvh, (0, math.nan, 0), 0.5)
        with self.assertRaises(ValueError):
            closest_point(bvh, (0, 0, math.inf), 0.5)
        with self.assertRaises(TypeError):
            closest_point(bvh, (0, 0, True), 0.5)

    def test_query_does_not_mutate_tree(self):
        rng = random.Random(11)
        tris = [_rand_tri(rng) for _ in range(30)]
        bvh = build(tris)
        before = closest_point(bvh, (0, 0, 0), 0.5)
        mb.intersect(bvh, mb.Ray((0, 0, -10), (0, 0, 1)), 0.5)
        after = closest_point(bvh, (0, 0, 0), 0.5)
        self.assertEqual(before, after)
        self.assertEqual(before.nodes_visited, after.nodes_visited)

    def test_module_and_method_entry_equivalent(self):
        tri = Triangle.stationary((0, 0, 0), (1, 0, 0), (0, 1, 0))
        bvh = build([tri])
        self.assertEqual(closest_point(bvh, (0.2, 0.2, 1.0), 0.5),
                         bvh.closest_point((0.2, 0.2, 1.0), 0.5))


class TestAgainstReference(unittest.TestCase):
    """随机场景与独立逐三角形投影参考比对。"""

    def test_random_scenes_match_reference(self):
        rng = random.Random(20260926)
        for scene in range(6):
            n = rng.randint(1, 40)
            tris = [_rand_tri(rng) for _ in range(n)]
            bvh = build(tris)
            for _ in range(15):
                p = (rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-6, 6))
                tau = rng.choice([0.0, 0.3, 0.5, 0.75, 1.0])
                exp = reference_closest_point(tris, p, tau)
                got = closest_point(bvh, p, tau)
                self.assertTrue(got.hit)
                self.assertEqual(got.face_id, exp["face_id"],
                                 (scene, p, tau))
                self.assertAlmostEqual(got.dist_sq, exp["dist_sq"],
                                       places=8, msg=(scene, p, tau))
                for k in range(3):
                    self.assertAlmostEqual(got.point[k], exp["point"][k],
                                           places=8, msg=(scene, p, tau))

    def test_pruning_on_scattered_grid(self):
        """分散网格上必须发生剪枝：实测面数远小于总面数。
        只断言计数关系（剪枝发生），不以速度数字作正确性证据。"""
        tris = _scattered_grid()
        n = len(tris)
        bvh = build(tris)
        rng = random.Random(3)
        total_tested = 0
        queries = 0
        for _ in range(30):
            p = (rng.uniform(-2, 50), rng.uniform(-2, 50), rng.uniform(-2, 20))
            tau = rng.random()
            got = closest_point(bvh, p, tau)
            exp = reference_closest_point(tris, p, tau)
            self.assertTrue(got.hit)
            self.assertEqual(got.face_id, exp["face_id"])
            self.assertAlmostEqual(got.dist_sq, exp["dist_sq"], places=8)
            total_tested += got.triangles_tested
            queries += 1
            # 单次查询不允许无条件测全部面
            self.assertLess(got.triangles_tested, n)
            self.assertGreater(got.triangles_tested, 0)
            self.assertGreater(got.nodes_visited, 0)
        avg = total_tested / queries
        # 平均实测面数显著低于全量（36 面），证明分支限界生效
        self.assertLess(avg, n / 3)


if __name__ == "__main__":
    unittest.main()

