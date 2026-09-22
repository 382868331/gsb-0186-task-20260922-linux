"""遍历正确性：与独立参考实现（tests/reference.py）逐射线比对。

覆盖：运动跨越、贴盒边/三角边/顶点、t 区间边界、退化面（含中途退化）、
同距面按 face_id、零方向分量、barycentric 精确值、计数与随机一致性。
独立参考不导入 motion_bvh，期望值不来自被测核心。
"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_bvh import Ray, build  # noqa: E402
from reference import reference_intersect  # noqa: E402


def _tri(a0, b0, c0, a1=None, b1=None, c1=None):
    """静态时两端相同。"""
    a1 = a0 if a1 is None else a1
    b1 = b0 if b1 is None else b1
    c1 = c0 if c1 is None else c1
    return ((a0, b0, c0), (a1, b1, c1))


def _agree(hit, expected, t_tol=1e-11, bary_tol=1e-9):
    if expected is None:
        return hit is None
    if hit is None:
        return False
    if hit.face_id != expected["face_id"]:
        return False
    scale = max(1.0, abs(expected["t"]))
    if abs(hit.t - expected["t"]) > t_tol * scale:
        return False
    for got, want in zip(hit.barycentric, expected["barycentric"]):
        if abs(got - want) > bary_tol:
            return False
    if abs(sum(hit.barycentric) - 1.0) > 1e-12:
        return False
    return True


def _compare_ray(testcase, scene, origin, direction, time, tmin=0.0, tmax=float("inf"),
                 t_tol=1e-11, bary_tol=1e-9):
    bvh = build(scene)
    ray = Ray(tuple(origin), tuple(direction), tmin=tmin, tmax=tmax)
    hit = bvh.intersect(ray, time)
    expected = reference_intersect(scene, tuple(origin), tuple(direction), time, tmin, tmax)
    testcase.assertTrue(
        _agree(hit, expected, t_tol, bary_tol),
        msg=(
            f"mismatch o={origin} d={direction} time={time} window=({tmin},{tmax})\n"
            f"  bvh={None if hit is None else (hit.face_id, hit.t, hit.barycentric)}\n"
            f"  ref={None if expected is None else (expected['face_id'], expected['t'], expected['barycentric'])}"
        ),
    )
    return hit, expected


class MotionAcrossTimesTests(unittest.TestCase):
    def test_triangle_moves_through_fixed_ray(self):
        # 三角形整体 z 从 -2（t=0）线性移动到 +2（t=1）；射线水平穿过 z=0。
        v0 = ((0.0, -1.0, -2.0), (2.0, -1.0, -2.0), (0.0, 1.0, -2.0))
        v1 = ((0.0, -1.0, 2.0), (2.0, -1.0, 2.0), (0.0, 1.0, 2.0))
        scene = [(v0, v1)]
        for time in (0.0, 0.1, 0.25, 0.49, 0.5, 0.51, 0.75, 0.9, 1.0):
            with self.subTest(time=time):
                _compare_ray(self, scene, (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), time)

    def test_endpoint_times_use_stored_vertices_exactly(self):
        # t=0/1 必须精确等于存储端点（插值公式在端点处无误差）。
        v0 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        v1 = ((5.0, 5.0, 1.0), (6.0, 5.0, 1.0), (5.0, 6.0, 1.0))
        scene = [(v0, v1)]
        for time, origin, distance in (
            (0.0, (0.2, 0.2, 5.0), 5.0),
            (1.0, (5.2, 5.2, 5.0), 4.0),
        ):
            hit, _ = _compare_ray(self, scene, origin, (0.0, 0.0, -1.0), time)
            self.assertEqual(hit.face_id, 0)
            self.assertAlmostEqual(hit.t, distance, places=12)

    def test_vertices_move_independently(self):
        # 三个顶点朝不同方向运动，面积随时间变化。
        v0 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        v1 = ((0.0, 0.0, 4.0), (0.0, -2.0, 0.0), (-3.0, 1.0, 0.0))
        scene = [(v0, v1)]
        for time in (0.0, 1 / 3, 0.5, 2 / 3, 1.0):
            for ox in (0.1, 0.3, -0.4):
                _compare_ray(self, scene, (ox, 0.2, 6.0), (0.05, -0.03, -1.0), time)


class EdgeAndWindowTests(unittest.TestCase):
    def test_hit_on_triangle_edges_and_vertices(self):
        scene = [_tri((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0))]
        # 边中点 (1.5,0)、斜边中点 (1.5,1.5)、顶点 a/b/c。
        for point in ((1.5, 0.0), (1.5, 1.5), (0.0, 0.0), (3.0, 0.0), (0.0, 3.0)):
            with self.subTest(point=point):
                hit, _ = _compare_ray(
                    self, scene, (point[0], point[1], 4.0), (0.0, 0.0, -1.0), 0.0,
                    bary_tol=1e-10,
                )
                self.assertAlmostEqual(hit.t, 4.0, places=11)

    def test_rays_along_grid_box_boundaries(self):
        # 整数网格让很多射线恰好贴着节点/三角形 AABB 的边界面；逐格与参考比对。
        scene = [
            _tri((float(i), float(j), 0.0), (float(i) + 1.0, float(j), 0.0),
                 (float(i), float(j) + 1.0, 0.0))
            for i in range(-4, 4)
            for j in range(-4, 4)
        ]
        # 网格整体沿 z 小幅运动，保证走运动界路径。
        moved = [
            (tri[0], tuple((x, y, z + 0.5) for (x, y, z) in tri[1]))
            for tri in scene
        ]
        for time in (0.0, 0.37, 1.0):
            for k in range(-4, 5):
                _compare_ray(self, moved, (float(k), float(k) + 0.27, 5.0),
                             (0.0, 0.0, -1.0), time)
                _compare_ray(self, moved, (float(k) + 0.27, float(k), 5.0),
                             (0.0, 0.0, -1.0), time)

    def test_hit_exactly_at_t_window_boundaries(self):
        scene = [_tri((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
        origin, direction = (0.2, 0.2, 5.0), (0.0, 0.0, -1.0)
        # 命中参数恰为 t=5。
        hit, _ = _compare_ray(self, scene, origin, direction, 0.0, tmin=5.0)
        self.assertAlmostEqual(hit.t, 5.0, places=12)
        hit, _ = _compare_ray(self, scene, origin, direction, 0.0, tmax=5.0)
        self.assertAlmostEqual(hit.t, 5.0, places=12)
        hit, _ = _compare_ray(self, scene, origin, direction, 0.0, tmin=5.0, tmax=5.0)
        self.assertIsNotNone(hit)
        # 区间外：双方都证明无命中。
        hit, expected = _compare_ray(self, scene, origin, direction, 0.0, tmin=5.0 + 1e-6)
        self.assertIsNone(hit)
        self.assertIsNone(expected)
        hit, expected = _compare_ray(self, scene, origin, direction, 0.0, tmax=5.0 - 1e-6)
        self.assertIsNone(hit)
        self.assertIsNone(expected)

    def test_grazing_ray_just_misses(self):
        # 射线在三角形边外侧极小距离：不能被 GUARD 错误“吸”成命中。
        scene = [_tri((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
        hit, expected = _compare_ray(
            self, scene, (-1e-7, 0.5, 2.0), (0.0, 0.0, -1.0), 0.0
        )
        self.assertIsNone(expected)
        self.assertIsNone(hit)


class DegenerateTests(unittest.TestCase):
    def test_zero_area_triangle_never_hits(self):
        scene = [
            _tri((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # 面 0：退化
            _tri((2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0)),  # 面 1：正常
        ]
        for origin in ((0.0, 0.5, 3.0), (0.0, 0.0, 3.0), (0.2, 0.2, 3.0)):
            hit, expected = _compare_ray(self, scene, origin, (0.0, 0.0, -1.0), 0.0)
        # 退化面区域射线：无命中。
        hit = build(scene).intersect(Ray((0.0, 0.5, 3.0), (0.0, 0.0, -1.0)), 0.0)
        self.assertIsNone(hit)
        # 旁边正常面仍可命中。
        hit = build(scene).intersect(Ray((2.2, 0.2, 3.0), (0.0, 0.0, -1.0)), 0.0)
        self.assertEqual(hit.face_id, 1)

    def test_point_triangle_never_hits(self):
        scene = [_tri((1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0))]
        hit, expected = _compare_ray(self, scene, (1.0, 1.0, 3.0), (0.0, 0.0, -1.0), 0.0)
        self.assertIsNone(hit)
        self.assertIsNone(expected)

    def test_triangle_collapses_at_mid_time(self):
        # a、b 在 t=0.5 恰好重合：该时刻面积为 0，必须 miss；其余时刻命中。
        v0 = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        v1 = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        scene = [(v0, v1)]
        for time, should_hit in ((0.0, True), (0.25, True), (0.5, False), (0.75, True), (1.0, True)):
            with self.subTest(time=time):
                hit, expected = _compare_ray(
                    self, scene, (0.0, 0.4, 3.0), (0.0, 0.0, -1.0), time
                )
                self.assertEqual(hit is not None, should_hit)
                self.assertEqual(expected is not None, should_hit)

    def test_degenerate_among_moving_mesh_fuzz(self):
        rng = random.Random(20260922)
        scene = []
        for i in range(30):
            if i % 4 == 0:
                # 恒退化：两个顶点始终相同。
                p = tuple(rng.uniform(-3, 3) for _ in range(3))
                q = tuple(rng.uniform(-3, 3) for _ in range(3))
                scene.append(((p, p, q), (p, p, q)))
            else:
                v0 = tuple(tuple(rng.uniform(-3, 3) for _ in range(3)) for _ in range(3))
                v1 = tuple(tuple(rng.uniform(-3, 3) for _ in range(3)) for _ in range(3))
                scene.append((v0, v1))
        for k in range(120):
            origin = tuple(rng.uniform(-9, 9) for _ in range(3))
            direction = tuple(rng.uniform(-1, 1) for _ in range(3))
            _compare_ray(self, scene, origin, direction, rng.random())


class TieDistanceTests(unittest.TestCase):
    def test_coincident_pair_prefers_smaller_face_id(self):
        t0 = _tri((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        t1 = _tri((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        for scene, winner in (([t0, t1], 0), ([t1, t0], 0)):
            hit, expected = _compare_ray(
                self, scene, (0.2, 0.2, 3.0), (0.0, 0.0, -1.0), 0.0
            )
            self.assertEqual(hit.face_id, winner)
            self.assertEqual(expected["face_id"], winner)

    def test_tie_pair_buried_among_other_faces(self):
        deco = [
            _tri((5.0, 5.0, 0.0), (6.0, 5.0, 0.0), (5.0, 6.0, 0.0)),
            _tri((-5.0, -5.0, 0.0), (-4.0, -5.0, 0.0), (-5.0, -4.0, 0.0)),
        ]
        pair = [
            _tri((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            _tri((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ]
        scene = [deco[0], pair[0], deco[1], pair[1]]  # 同距面 id 1 与 3
        hit, expected = _compare_ray(
            self, scene, (0.3, 0.3, 2.0), (0.0, 0.0, -1.0), 0.0
        )
        self.assertEqual(hit.face_id, 1)
        self.assertEqual(expected["face_id"], 1)
        # 反向射线规则一致。
        hit, expected = _compare_ray(
            self, scene, (0.3, 0.3, -2.0), (0.0, 0.0, 1.0), 0.0
        )
        self.assertEqual(hit.face_id, 1)
        self.assertEqual(expected["face_id"], 1)

    def test_nearbutnot_equal_distance_not_a_tie(self):
        # 距离差 1e-4 远超同距带：射线从 z=3 向下，z=0 的面（t=3）严格近于
        # z=-0.0001 的面（t=3.0001），与 face_id 顺序无关。
        closer = _tri((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        behind = (
            ((0.0, 0.0, -0.0001), (1.0, 0.0, -0.0001), (0.0, 1.0, -0.0001)),
            ((0.0, 0.0, -0.0001), (1.0, 0.0, -0.0001), (0.0, 1.0, -0.0001)),
        )
        hit, _ = _compare_ray(self, [behind, closer], (0.2, 0.2, 3.0), (0.0, 0.0, -1.0), 0.0)
        self.assertEqual(hit.face_id, 1)
        hit, _ = _compare_ray(self, [closer, behind], (0.2, 0.2, 3.0), (0.0, 0.0, -1.0), 0.0)
        self.assertEqual(hit.face_id, 0)
        self.assertAlmostEqual(hit.t, 3.0, places=9)


class ZeroDirectionTests(unittest.TestCase):
    def test_axis_aligned_directions(self):
        scene = [
            _tri((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            (
                ((0.0, 0.0, 2.0), (0.0, 1.0, 2.0), (0.0, 0.0, 3.0)),
                ((1.0, 0.0, 2.0), (1.0, 1.0, 2.0), (1.0, 0.0, 3.0)),
            ),
        ]
        rays = [
            ((0.2, 0.2, -3.0), (0.0, 0.0, 1.0)),
            ((-3.0, 0.3, 2.4), (1.0, 0.0, 0.0)),
            ((3.0, 0.3, 2.4), (-1.0, 0.0, 0.0)),
            ((0.2, -3.0, 0.2), (0.0, 1.0, 0.0)),
            ((0.2, 3.0, 0.2), (0.0, -1.0, 0.0)),
            ((0.2, 0.2, 3.0), (0.0, 0.0, -1.0)),
        ]
        for time in (0.0, 0.5, 1.0):
            for origin, direction in rays:
                _compare_ray(self, scene, origin, direction, time)

    def test_ray_parallel_to_triangle_plane_misses(self):
        scene = [_tri((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
        hit, expected = _compare_ray(
            self, scene, (0.2, 0.2, 1.0), (1.0, 0.0, 0.0), 0.0
        )
        self.assertIsNone(hit)
        self.assertIsNone(expected)


class ExactBarycentricTests(unittest.TestCase):
    def test_centroid_edges_vertices_exact_values(self):
        scene = [_tri((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0))]
        bvh = build(scene)
        cases = [
            ((1.0, 1.0), (1 / 3, 1 / 3, 1 / 3)),
            ((1.5, 0.0), (0.5, 0.5, 0.0)),
            ((1.5, 1.5), (0.0, 0.5, 0.5)),
            ((0.0, 3.0), (0.0, 0.0, 1.0)),
        ]
        for (x, y), want in cases:
            hit = bvh.intersect(Ray((x, y, 2.0), (0.0, 0.0, -1.0)), 0.0)
            for got, w in zip(hit.barycentric, want):
                self.assertAlmostEqual(got, w, places=12, msg=f"point={(x, y)}")

    def test_bary_moves_with_time(self):
        # 命中点固定为世界点 (0.5,0.5)，三角形在运动；bary 必须随时间变化且与参考一致。
        v0 = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))
        v1 = ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))
        scene = [(v0, v1)]
        last = None
        for time in (0.0, 0.25, 0.5, 0.75):
            hit, expected = _compare_ray(
                self, scene, (0.5, 0.5, 4.0), (0.0, 0.0, -1.0), time
            )
            self.assertIsNotNone(hit)
            self.assertTrue(last is None or hit.barycentric != last)
            last = hit.barycentric


class CounterTests(unittest.TestCase):
    def test_culled_miss_tests_zero_triangles(self):
        scene = [
            _tri((float(i), 0.0, 0.0), (float(i) + 0.5, 0.0, 0.0),
                 (float(i), 0.5, 0.0))
            for i in range(32)
        ]
        bvh = build(scene)
        # 射线与三角形平面平行且不共面：无命中（计数经 intersect_full 取得）。
        parallel_hit, parallel_stats = bvh.intersect_full(
            Ray((0.25, 100.0, 0.25), (0.0, -1.0, 0.0)), 0.0
        )
        self.assertIsNone(parallel_hit)
        self.assertEqual(parallel_stats.triangle_tests, 0)
        # 射线从场景外侧出发：根节点即被剔除。
        miss, stats = bvh.intersect_full(Ray((100.0, 100.0, 100.0), (1.0, 0.0, 0.0)), 0.0)
        self.assertIsNone(miss)
        self.assertEqual(stats.triangle_tests, 0)
        self.assertEqual(stats.nodes_visited, 1)

    def test_triangle_tests_never_exceeds_scene_size(self):
        rng = random.Random(4242)
        scene = [
            (
                tuple(tuple(rng.uniform(-4, 4) for _ in range(3)) for _ in range(3)),
                tuple(tuple(rng.uniform(-4, 4) for _ in range(3)) for _ in range(3)),
            )
            for _ in range(60)
        ]
        bvh = build(scene)
        for _ in range(200):
            ray = Ray(
                tuple(rng.uniform(-10, 10) for _ in range(3)),
                tuple(rng.uniform(-1, 1) for _ in range(3)),
            )
            hit = bvh.intersect(ray, rng.random())
            if hit is not None:
                self.assertLessEqual(hit.triangle_tests, 60)
                self.assertGreaterEqual(hit.triangle_tests, 1)


class FuzzReferenceTests(unittest.TestCase):
    def test_random_scenes_match_reference(self):
        rng = random.Random(987654321)
        for scene_index in range(8):
            n = rng.randint(1, 80)
            scene = []
            for _ in range(n):
                v0 = tuple(tuple(rng.uniform(-4, 4) for _ in range(3)) for _ in range(3))
                v1 = tuple(tuple(rng.uniform(-4, 4) for _ in range(3)) for _ in range(3))
                scene.append((v0, v1))
            bvh = build(scene)
            for _ in range(60):
                origin = tuple(rng.uniform(-9, 9) for _ in range(3))
                direction = tuple(rng.uniform(-1, 1) for _ in range(3))
                tmax = float("inf") if rng.random() < 0.8 else rng.uniform(1.0, 20.0)
                tmin = 0.0 if rng.random() < 0.8 else rng.uniform(-1.0, 1.0)
                _compare_ray(
                    self, scene, origin, direction, rng.random(),
                    tmin=tmin, tmax=tmax,
                )

    def test_bvh_tests_fewer_triangles_than_brute_force(self):
        # 64 个彼此分离的小三角形：遍历只测射线附近的叶子。
        scene = [
            (
                ((float(i), 0.0, 0.0), (float(i) + 0.4, 0.0, 0.0),
                 (float(i), 0.0, 0.4)),
                ((float(i), 2.0, 1.0), (float(i) + 0.4, 2.0, 1.0),
                 (float(i), 2.0, 1.4)),
            )
            for i in range(64)
        ]
        bvh = build(scene)
        rng = random.Random(1357)
        total_tests = 0
        rays = 200
        for _ in range(rays):
            x = rng.uniform(0, 64)
            hit = bvh.intersect(
                Ray((x, -1.0, 0.2), (0.0, 1.0, 0.05 * rng.uniform(-1, 1))),
                rng.random(),
            )
            total_tests += hit.triangle_tests if hit else 0
        # 暴力法每射线必测 64；BVH 平均必须显著更少。
        self.assertLess(total_tests / rays, 16.0)


if __name__ == "__main__":
    unittest.main()
