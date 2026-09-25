"""packet_occluded 批量遮挡查询的必要测试。

期望值来源：tests/reference.py 中独立书写的逐面遮挡 oracle
（reference_occluded，不调用 motion_bvh 的任何算法）。
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
from motion_bvh import (
    Triangle, Ray, OcclusionQuery, build, intersect, packet_occluded,
)
from reference import reference_occluded


# ---------------------------------------------------------------------------
# 确定性数据生成（与 test_motion_bvh 同风格，本地独立书写）
# ---------------------------------------------------------------------------

def _rand_tri(rng, motion_scale=2.0):
    def pt():
        return (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-4, 4))

    a0, b0, c0 = pt(), pt(), pt()
    mv = [tuple(rng.uniform(-motion_scale, motion_scale) for _ in range(3))
          for _ in range(3)]
    a1 = tuple(a0[k] + mv[0][k] for k in range(3))
    b1 = tuple(b0[k] + mv[1][k] for k in range(3))
    c1 = tuple(c0[k] + mv[2][k] for k in range(3))
    return Triangle(a0, b0, c0, a1, b1, c1)


def _wild_ray(rng):
    return Ray(
        tuple(rng.uniform(-20, 20) for _ in range(3)),
        tuple(rng.uniform(-1, 1) for _ in range(3)),
    )


def _aimed_ray(rng, tris, tau):
    tri = rng.choice(tris)
    p0 = tri.a0
    p1 = tri.a1
    target = tuple(p0[k] + tau * (p1[k] - p0[k]) for k in range(3))
    origin = tuple(target[k] + rng.uniform(-10, 10) for k in range(3))
    d = tuple(target[k] - origin[k] for k in range(3))
    if rng.random() < 0.3:
        j = rng.randrange(3)
        d = list(d)
        d[j] += rng.uniform(-1.0, 1.0)
        d = tuple(d)
    return Ray(origin, d)


def _two_clusters():
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


def _big_tri(z, face_id=None):
    return Triangle.stationary(
        (-2.0, -2.0, z), (2.0, -2.0, z), (0.0, 2.0, z), face_id=face_id)


# ---------------------------------------------------------------------------
# 布尔正确性：与独立逐面 oracle 核对
# ---------------------------------------------------------------------------

class TestPacketVsOracle(unittest.TestCase):
    def test_fuzz_packets(self):
        rng = random.Random(20260925)
        checked = 0
        for n in (1, 9, 40, 120, 300):
            tris = [_rand_tri(rng) for _ in range(n)]
            bvh = build(tris)
            fids = list(range(n))
            for _ in range(25):
                k = rng.randrange(1, 33)
                queries = []
                for _ in range(k):
                    tau = rng.choice((0.0, 0.33, 0.5, 0.9, 1.0))
                    ray = (_aimed_ray(rng, tris, tau)
                           if rng.random() < 0.7 else _wild_ray(rng))
                    t0 = rng.choice((0.0, 0.5, 2.0, -1.0))
                    t1 = rng.choice((math.inf, t0 + 1.5, t0))  # 含空区间
                    if rng.random() < 0.5:
                        ig = frozenset(rng.sample(
                            fids, rng.randrange(0, min(4, len(fids)) + 1)))
                    else:
                        ig = frozenset()
                    queries.append(OcclusionQuery(
                        ray, time=tau, t_min=t0, t_max=t1, ignore_faces=ig))
                res = packet_occluded(bvh, queries)
                self.assertEqual(len(res.occluded), k)
                for i, q in enumerate(queries):
                    exp = reference_occluded(
                        tris, q.ray, q.time, q.t_min, q.t_max, q.ignore_faces)
                    self.assertEqual(
                        res.occluded[i], exp,
                        f"lane {i} 不一致: ray={q.ray} τ={q.time} "
                        f"[{q.t_min},{q.t_max}) ignore={q.ignore_faces}")
                    checked += 1
        self.assertGreater(checked, 1000)

    def test_default_window_matches_single_intersect(self):
        # 默认窗口 [0, +inf) 下，遮挡布尔与旧单射线 intersect.hit 一致
        rng = random.Random(77)
        tris = [_rand_tri(rng) for _ in range(60)]
        bvh = build(tris)
        rays, taus = [], []
        for _ in range(40):
            tau = rng.choice((0.0, 0.5, 1.0))
            rays.append(_aimed_ray(rng, tris, tau) if rng.random() < 0.7
                        else _wild_ray(rng))
            taus.append(tau)
        res = packet_occluded(
            bvh, [OcclusionQuery(r, time=t) for r, t in zip(rays, taus)])
        for i, (r, tau) in enumerate(zip(rays, taus)):
            self.assertEqual(res.occluded[i], intersect(bvh, r, tau).hit)

    def test_scattered_rays(self):
        # 完全分散的射线：方向/原点无相关性，仍逐条与 oracle 一致
        rng = random.Random(8)
        tris = _two_clusters()
        bvh = build(tris)
        rays = [_wild_ray(rng) for _ in range(24)]
        res = packet_occluded(bvh, rays)  # 裸 Ray 走默认参数
        for i, r in enumerate(rays):
            self.assertEqual(res.occluded[i],
                             reference_occluded(tris, r, 0.0))


class TestPacketSemantics(unittest.TestCase):
    def test_t_max_touch_is_not_occlusion(self):
        tri = _big_tri(5.0)
        bvh = build([tri])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        # t == t_max 恰好接触：不算遮挡
        res = packet_occluded(bvh, [OcclusionQuery(ray, t_max=5.0)])
        self.assertEqual(res.occluded, (False,))
        # t_max 稍大：算遮挡
        res = packet_occluded(bvh, [OcclusionQuery(ray, t_max=5.0 + 1e-9)])
        self.assertEqual(res.occluded, (True,))
        # t == t_min 算入（半开区间左闭）
        res = packet_occluded(
            bvh, [OcclusionQuery(ray, t_min=5.0, t_max=6.0)])
        self.assertEqual(res.occluded, (True,))
        # t_min == t_max：空区间，恒不遮挡
        res = packet_occluded(
            bvh, [OcclusionQuery(ray, t_min=5.0, t_max=5.0)])
        self.assertEqual(res.occluded, (False,))
        self.assertEqual(res.nodes_visited, 0)  # 空区间不参与调度

    def test_ignore_near_face_then_far_face_occludes(self):
        near = _big_tri(2.0, face_id=10)
        far = _big_tri(4.0, face_id=20)
        tris = [near, far]
        bvh = build(tris)
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        # 忽略近面 → 远面仍然遮挡
        res = packet_occluded(
            bvh, [OcclusionQuery(ray, ignore_faces={10})])
        self.assertEqual(res.occluded, (True,))
        self.assertTrue(reference_occluded(tris, ray, 0.0, ignore_faces={10}))
        # 近远都忽略 → 不遮挡
        res = packet_occluded(
            bvh, [OcclusionQuery(ray, ignore_faces={10, 20})])
        self.assertEqual(res.occluded, (False,))

    def test_ignore_faces_only_affects_own_ray(self):
        tri = _big_tri(3.0, face_id=7)
        bvh = build([tri])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        # 两条完全相同的射线：一条忽略唯一遮挡面，一条不忽略
        res = packet_occluded(bvh, [
            OcclusionQuery(ray, ignore_faces={7}),
            OcclusionQuery(ray),
        ])
        self.assertEqual(res.occluded, (False, True))

    def test_zero_direction_components(self):
        cases = [
            ((0.0, 0.0, 1.0), _big_tri(5.0)),
            ((0.0, 1.0, 0.0), Triangle.stationary(
                (-1.0, 5.0, -1.0), (1.0, 5.0, -1.0), (0.0, 5.0, 1.0))),
            ((1.0, 0.0, 0.0), Triangle.stationary(
                (5.0, -1.0, -1.0), (5.0, 1.0, -1.0), (5.0, 0.0, 1.0))),
        ]
        tris = [t for _, t in cases]
        bvh = build(tris)
        queries = [OcclusionQuery(Ray((0.0, 0.0, 0.0), d)) for d, _ in cases]
        # 再加一条零分量但原点在板外的射线（x 方向、y 偏移出板）
        queries.append(OcclusionQuery(Ray((0.0, 9.0, 0.0), (1.0, 0.0, 0.0))))
        res = packet_occluded(bvh, queries)
        self.assertEqual(res.occluded, (True, True, True, False))
        for i, q in enumerate(queries):
            self.assertEqual(
                res.occluded[i],
                reference_occluded(tris, q.ray, 0.0), f"lane {i}")

    def test_different_times_per_ray(self):
        # 三角形 t=0 在 x=0，t=1 移到 x=10；同形射线不同 time 结果不同
        a0, b0, c0 = (-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0)
        tri = Triangle(a0, b0, c0,
                       (a0[0] + 10, a0[1], a0[2]),
                       (b0[0] + 10, b0[1], b0[2]),
                       (c0[0] + 10, c0[1], c0[2]))
        bvh = build([tri])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = packet_occluded(bvh, [
            OcclusionQuery(ray, time=0.0),
            OcclusionQuery(ray, time=1.0),
            OcclusionQuery(ray, time=0.5),
        ])
        self.assertEqual(res.occluded, (True, False, False))
        for i, tau in enumerate((0.0, 1.0, 0.5)):
            self.assertEqual(res.occluded[i],
                             reference_occluded([tri], ray, tau))

    def test_early_exit_does_not_clear_other_lanes(self):
        # 两条射线分别只命中不同子树；先解决的 lane 不得误清另一条
        tris = []
        for i in range(12):
            tris.append(Triangle.stationary(
                (-10.0 + i * 0.5, 0.0, 5.0), (-9.6 + i * 0.5, 0.0, 5.0),
                (-9.8 + i * 0.5, 1.0, 5.0)))
        for i in range(12):
            tris.append(Triangle.stationary(
                (4.0 + i * 0.5, 0.0, 5.0), (4.4 + i * 0.5, 0.0, 5.0),
                (4.2 + i * 0.5, 1.0, 5.0)))
        bvh = build(tris)
        ray_l = Ray((-9.8, 0.3, 0.0), (0.0, 0.0, 1.0))
        ray_r = Ray((4.2, 0.3, 0.0), (0.0, 0.0, 1.0))
        ray_miss = Ray((0.0, 30.0, 0.0), (0.0, 0.0, 1.0))
        res = packet_occluded(bvh, [ray_l, ray_r, ray_miss])
        self.assertEqual(res.occluded, (True, True, False))
        for i, r in enumerate((ray_l, ray_r, ray_miss)):
            self.assertEqual(res.occluded[i],
                             reference_occluded(tris, r, 0.0))

    def test_empty_packet_and_empty_bvh(self):
        bvh = build([_big_tri(5.0)])
        res = packet_occluded(bvh, [])
        self.assertEqual(res.occluded, ())
        self.assertEqual(res.nodes_visited, 0)
        self.assertEqual(res.box_tests, 0)
        self.assertEqual(res.triangle_tests, 0)
        empty = build([])
        res = packet_occluded(empty, [Ray((0, 0, 0), (0, 0, 1))])
        self.assertEqual(res.occluded, (False,))
        self.assertEqual(res.nodes_visited, 0)


# ---------------------------------------------------------------------------
# 共享调度与计数
# ---------------------------------------------------------------------------

class TestSharedScheduling(unittest.TestCase):
    def test_coherent_bundle_fewer_nodes_than_per_ray(self):
        tris = _two_clusters()
        bvh = build(tris)
        rng = random.Random(5)
        rays = []
        for _ in range(12):
            rays.append(Ray(
                (6.2 + rng.uniform(-0.05, 0.05),
                 0.3333333333333333 + rng.uniform(-0.05, 0.05), 0.0),
                (0.0, 0.0, 1.0)))
        res = packet_occluded(bvh, rays)
        singles = [intersect(bvh, r, 0.0) for r in rays]
        total_single_nodes = sum(s.nodes_visited for s in singles)
        # 共享调度：整批节点调度次数严格少于逐条遍历之和
        self.assertLess(res.nodes_visited, total_single_nodes)
        # 同向成束：整批调度不超过任一单条的节点数（同一棵树上走一遍）
        self.assertLessEqual(res.nodes_visited,
                             max(s.nodes_visited for s in singles))
        for i, s in enumerate(singles):
            self.assertEqual(res.occluded[i], s.hit)

    def test_identical_rays_share_exactly(self):
        # 8 条完全相同的射线：节点调度与单条同量级，盒测试按 lane 计
        tris = _two_clusters()
        bvh = build(tris)
        ray = Ray((6.2, 0.3333333333333333, 0.0), (0.0, 0.0, 1.0))
        single = intersect(bvh, ray, 0.0)
        res = packet_occluded(bvh, [ray] * 8)
        self.assertEqual(res.occluded, (single.hit,) * 8)
        self.assertLessEqual(res.nodes_visited, single.nodes_visited)
        # 每条存活射线在每个被调度节点恰好一次盒测试
        self.assertEqual(res.box_tests, 8 * res.nodes_visited)
        self.assertGreater(res.triangle_tests, 0)

    def test_counts_are_separate_and_sane(self):
        tris = _two_clusters()
        bvh = build(tris)
        # 全部射线明显在场景外：根盒即拒绝
        rays = [Ray((1000.0, 1000.0, 0.0), (0.0, 0.0, 1.0)) for _ in range(6)]
        res = packet_occluded(bvh, rays)
        self.assertEqual(res.occluded, (False,) * 6)
        self.assertEqual(res.nodes_visited, 1)   # 只调度根
        self.assertEqual(res.box_tests, 6)       # 每 lane 一次根盒测试
        self.assertEqual(res.triangle_tests, 0)

    def test_ignore_faces_reduces_triangle_tests(self):
        tris = _two_clusters()
        bvh = build(tris)
        ray = Ray((6.2, 0.3333333333333333, 0.0), (0.0, 0.0, 1.0))
        all_ids = frozenset(range(len(tris)))
        res_plain = packet_occluded(bvh, [OcclusionQuery(ray)])
        res_ign = packet_occluded(
            bvh, [OcclusionQuery(ray, ignore_faces=all_ids)])
        self.assertEqual(res_ign.occluded, (False,))
        self.assertEqual(res_ign.triangle_tests, 0)  # 全部忽略：不测试
        self.assertGreater(res_plain.triangle_tests, 0)


# ---------------------------------------------------------------------------
# 整批拒绝（任一非法射线 → 整批抛错，无部分结果）
# ---------------------------------------------------------------------------

class TestPacketValidation(unittest.TestCase):
    def _ok_query(self):
        return OcclusionQuery(Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))

    def setUp(self):
        self.bvh = build([_big_tri(5.0)])

    def test_reject_over_64_rays(self):
        queries = [self._ok_query() for _ in range(mb.MAX_PACKET_RAYS + 1)]
        with self.assertRaises(ValueError):
            packet_occluded(self.bvh, queries)
        # 恰好 64 条合法
        res = packet_occluded(
            self.bvh, [self._ok_query() for _ in range(mb.MAX_PACKET_RAYS)])
        self.assertEqual(len(res.occluded), mb.MAX_PACKET_RAYS)

    def test_one_bad_ray_rejects_whole_batch(self):
        queries = [self._ok_query() for _ in range(4)]
        queries[2] = OcclusionQuery(Ray((0, 0, 0), (0.0, 0.0, 0.0)))
        with self.assertRaises(ValueError) as ctx:
            packet_occluded(self.bvh, queries)
        self.assertIn("queries[2]", str(ctx.exception))

    def test_reject_nan_inf_and_bool(self):
        with self.assertRaises(ValueError):
            packet_occluded(self.bvh, [OcclusionQuery(
                Ray((math.nan, 0, 0), (0, 0, 1)))])
        with self.assertRaises(ValueError):
            packet_occluded(self.bvh, [OcclusionQuery(
                Ray((0, 0, 0), (0, 0, 1)), time=math.inf)])
        with self.assertRaises(TypeError):
            packet_occluded(self.bvh, [OcclusionQuery(
                Ray((0, 0, 0), (0, 0, 1)), time=True)])
        with self.assertRaises(ValueError):
            packet_occluded(self.bvh, [OcclusionQuery(
                Ray((0, 0, 0), (0, 0, 1)), t_max=-math.inf)])
        with self.assertRaises(ValueError):
            packet_occluded(self.bvh, [OcclusionQuery(
                Ray((0, 0, 0), (0, 0, 1)), t_max=math.nan)])

    def test_reject_domain_errors(self):
        with self.assertRaises(ValueError):  # time 越界
            packet_occluded(self.bvh, [OcclusionQuery(
                Ray((0, 0, 0), (0, 0, 1)), time=1.5)])
        with self.assertRaises(ValueError):  # t_min > t_max
            packet_occluded(self.bvh, [OcclusionQuery(
                Ray((0, 0, 0), (0, 0, 1)), t_min=3.0, t_max=2.0)])
        with self.assertRaises(ValueError):  # 负 ignore id
            packet_occluded(self.bvh, [OcclusionQuery(
                Ray((0, 0, 0), (0, 0, 1)), ignore_faces={-1})])

    def test_reject_type_errors(self):
        with self.assertRaises(TypeError):
            packet_occluded("bvh", [self._ok_query()])
        with self.assertRaises(TypeError):
            packet_occluded(self.bvh, ["not a ray"])
        with self.assertRaises(TypeError):
            packet_occluded(self.bvh, [OcclusionQuery("not a ray")])
        with self.assertRaises(TypeError):  # ignore_faces 必须是集合
            packet_occluded(self.bvh, [OcclusionQuery(
                Ray((0, 0, 0), (0, 0, 1)), ignore_faces=[1, 2])])
        with self.assertRaises(TypeError):  # 元素拒绝 bool
            packet_occluded(self.bvh, [OcclusionQuery(
                Ray((0, 0, 0), (0, 0, 1)), ignore_faces={True})])
        with self.assertRaises(TypeError):
            packet_occluded(self.bvh, 42)

    def test_error_position_is_locatable(self):
        queries = [self._ok_query() for _ in range(5)]
        queries[3] = OcclusionQuery(
            Ray((0, 0, 0), (0, 0, 1)), time=math.nan)
        try:
            packet_occluded(self.bvh, queries)
        except ValueError as exc:
            self.assertIn("queries[3]", str(exc))
        else:
            self.fail("应抛出 ValueError")

    def test_bvh_method_and_module_function_equivalent(self):
        q = [OcclusionQuery(Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))]
        a = self.bvh.packet_occluded(q)
        b = mb.packet_occluded(self.bvh, q)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
