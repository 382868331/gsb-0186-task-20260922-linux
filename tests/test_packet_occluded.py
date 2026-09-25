"""packet_occluded 批量遮挡（阴影射线包）必要测试。

期望值来源：tests/reference.py 中独立书写的逐面暴力遮挡参考
（reference_occluded，浮点；reference_packet_occluded_exact，Fraction
精确有理数、零容差），均不调用 motion_bvh 的任何遍历/求交实现。
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
    Triangle, Ray, ShadowRay, OcclusionResult,
    build, intersect, packet_occluded,
)
from reference import reference_occluded, reference_packet_occluded_exact


# ---------------------------------------------------------------------------
# 确定性数据生成（与 test_motion_bvh 同风格，本文件自包含）
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
    tri = rng.choice(tris)
    target = _centroid_at(tri, tau)
    angle = rng.random() * 2 * math.pi
    radius = rng.uniform(6.0, 12.0)
    origin = (target[0] + radius * math.cos(angle),
              target[1] + radius * math.sin(angle),
              target[2] + rng.uniform(4.0, 10.0))
    d = tuple(target[k] - origin[k] for k in range(3))
    if rng.random() < 0.35:
        j = rng.randrange(3)
        d = list(d)
        d[j] += rng.uniform(-1.5, 1.5)
        d = tuple(d)
    return origin, d


def _wild_ray(rng):
    return (tuple(rng.uniform(-20, 20) for _ in range(3)),
            tuple(rng.uniform(-1, 1) for _ in range(3)))


def _wall_scene(n_per_wall=36):
    """z=±20 两堵三角形墙（静止），供成束射线共享调度测试。"""
    tris = []
    for z in (20.0, -20.0):
        for i in range(n_per_wall):
            gx = (i % 6) * 1.1 - 2.75
            gy = (i // 6) * 1.1 - 2.75
            s = 0.5
            tris.append(Triangle.stationary(
                (gx - s, gy - s, z), (gx + s, gy - s, z), (gx, gy + s, z)))
    return tris


# ---------------------------------------------------------------------------
# 基本契约
# ---------------------------------------------------------------------------

class TestPacketBasics(unittest.TestCase):
    def test_results_follow_input_order(self):
        tri = Triangle.stationary((-1, -1, 5), (1, -1, 5), (0, 1, 5))
        bvh = build([tri])
        hit = ShadowRay((0, 0, 0), (0, 0, 1))
        miss = ShadowRay((50, 50, 50), (0, 0, 1))
        res = packet_occluded(bvh, [miss, hit, miss, hit])
        self.assertIsInstance(res, OcclusionResult)
        self.assertEqual(res.occluded, (False, True, False, True))
        self.assertEqual(res.rejection, (None,) * 4)

    def test_accepts_tuple_and_generator(self):
        tri = Triangle.stationary((-1, -1, 5), (1, -1, 5), (0, 1, 5))
        bvh = build([tri])
        rays = (ShadowRay((0, 0, 0), (0, 0, 1)), ShadowRay((9, 9, 9), (1, 0, 0)))
        self.assertEqual(packet_occluded(bvh, rays).occluded, (True, False))
        gen = (sr for sr in rays)
        self.assertEqual(bvh.packet_occluded(gen).occluded, (True, False))

    def test_empty_bvh(self):
        bvh = build([])
        res = packet_occluded(bvh, [ShadowRay((0, 0, 0), (0, 0, 1)),
                                    ShadowRay((1, 2, 3), (1, 0, 0), time=1.0)])
        self.assertEqual(res.occluded, (False, False))
        self.assertEqual(res.nodes_scheduled, 0)
        self.assertEqual(res.box_tests, 0)
        self.assertEqual(res.triangle_tests, 0)

    def test_single_ray_packet_consistent_with_intersect(self):
        # 非边界窗口（t_min=0, t_max=inf）下，单条 packet 与旧单射线最近命中
        # 的“是否命中”判定必须一致（旧接口行为由既有 41 项回归保持）。
        rng = random.Random(17)
        tris = [_rand_tri(rng) for _ in range(50)]
        bvh = build(tris)
        for _ in range(40):
            o = tuple(rng.uniform(-10, 10) for _ in range(3))
            d = tuple(rng.uniform(-1, 1) for _ in range(3))
            tau = rng.choice((0.0, 0.5, 1.0))
            got = packet_occluded(bvh, [ShadowRay(o, d, time=tau)]).occluded[0]
            exp = intersect(bvh, Ray(o, d), tau).hit
            self.assertEqual(got, exp)

    def test_old_single_ray_nearest_hit_preserved(self):
        # 调用 packet 之后，旧单射线最近命中结果不变（packet 为只读查询）
        near = Triangle.stationary((-1, -1, 2), (1, -1, 2), (0, 1, 2), face_id=9)
        far = Triangle.stationary((-1, -1, 6), (1, -1, 6), (0, 1, 6), face_id=3)
        bvh = build([near, far])
        before = intersect(bvh, Ray((0, 0, 0), (0, 0, 1)), 0.0)
        packet_occluded(bvh, [ShadowRay((0, 0, 0), (0, 0, 1))] * 8)
        after = intersect(bvh, Ray((0, 0, 0), (0, 0, 1)), 0.0)
        self.assertTrue(after.hit)
        self.assertEqual(after.face_id, 9)  # 最近命中（不是 face_id 最小者）
        self.assertEqual((before.face_id, before.t), (after.face_id, after.t))


# ---------------------------------------------------------------------------
# 半开区间 [t_min, t_max)
# ---------------------------------------------------------------------------

class TestHalfOpenInterval(unittest.TestCase):
    def setUp(self):
        self.tri = Triangle.stationary((-1, -1, 5), (1, -1, 5), (0, 1, 5))
        self.bvh = build([self.tri])

    def test_t_max_touch_is_not_occlusion(self):
        res = packet_occluded(self.bvh, [ShadowRay((0, 0, 0), (0, 0, 1), t_max=5.0)])
        self.assertEqual(res.occluded, (False,))  # t==t_max 接触不算遮挡
        res2 = packet_occluded(
            self.bvh, [ShadowRay((0, 0, 0), (0, 0, 1), t_max=5.0 + 1e-9)])
        self.assertEqual(res2.occluded, (True,))

    def test_t_min_touch_is_occlusion(self):
        res = packet_occluded(
            self.bvh, [ShadowRay((0, 0, 0), (0, 0, 1), t_min=5.0, t_max=6.0)])
        self.assertEqual(res.occluded, (True,))  # t==t_min 算遮挡

    def test_empty_interval_never_occludes(self):
        res = packet_occluded(
            self.bvh, [ShadowRay((0, 0, 0), (0, 0, 1), t_min=5.0, t_max=5.0)])
        self.assertEqual(res.occluded, (False,))

    def test_window_clipping(self):
        self.assertEqual(
            packet_occluded(self.bvh, [ShadowRay((0, 0, 0), (0, 0, 1),
                                                 t_min=6.0)]).occluded, (False,))
        self.assertEqual(
            packet_occluded(self.bvh, [ShadowRay((0, 0, 0), (0, 0, 1),
                                                 t_max=4.0)]).occluded, (False,))
        self.assertEqual(
            packet_occluded(self.bvh, [ShadowRay((0, 0, 0), (0, 0, 1),
                                                 t_min=4.0, t_max=6.0)]).occluded,
            (True,))


# ---------------------------------------------------------------------------
# 忽略面集合（仅所属射线生效）
# ---------------------------------------------------------------------------

class TestIgnoreFaces(unittest.TestCase):
    def _two_walls(self):
        near = Triangle.stationary((-2, -2, 2), (2, -2, 2), (0, 2, 2), face_id=10)
        far = Triangle.stationary((-2, -2, 8), (2, -2, 8), (0, 2, 8), face_id=20)
        return [near, far]

    def test_ignore_near_face_hits_far_face(self):
        tris = self._two_walls()
        bvh = build(tris)
        rays = [ShadowRay((0, 0, 0), (0, 0, 1)),                        # 近面挡
                ShadowRay((0, 0, 0), (0, 0, 1), ignore_faces={10}),     # 忽略近面→远面挡
                ShadowRay((0, 0, 0), (0, 0, 1), ignore_faces={10, 20}),  # 全忽略→不挡
                ShadowRay((0, 0, 0), (0, 0, 1), ignore_faces={20})]     # 忽略远面→近面挡
        res = packet_occluded(bvh, rays)
        self.assertEqual(res.occluded, (True, True, False, True))
        for i, sr in enumerate(rays):
            self.assertEqual(
                res.occluded[i],
                reference_occluded(tris, sr.origin, sr.direction, 0.0,
                                   0.0, math.inf, sr.ignore_faces),
                f"lane {i}")

    def test_ignore_set_is_per_ray(self):
        # 两条完全相同的射线同批：一条忽略命中面、一条不忽略，互不影响
        tris = self._two_walls()
        bvh = build(tris)
        res = packet_occluded(bvh, [
            ShadowRay((0, 0, 0), (0, 0, 1), ignore_faces=(10, 20)),
            ShadowRay((0, 0, 0), (0, 0, 1))])
        self.assertEqual(res.occluded, (False, True))

    def test_unknown_ignore_id_rejected(self):
        bvh = build(self._two_walls())
        with self.assertRaises(ValueError):
            packet_occluded(bvh, [ShadowRay((0, 0, 0), (0, 0, 1),
                                            ignore_faces={999})])

    def test_ignore_faces_type_and_value_checks(self):
        bvh = build(self._two_walls())
        with self.assertRaises(TypeError):  # bool 不是 int
            packet_occluded(bvh, [ShadowRay((0, 0, 0), (0, 0, 1),
                                            ignore_faces={True})])
        with self.assertRaises(ValueError):  # 负 id
            packet_occluded(bvh, [ShadowRay((0, 0, 0), (0, 0, 1),
                                            ignore_faces={-1})])
        with self.assertRaises(TypeError):  # 非容器
            packet_occluded(bvh, [ShadowRay((0, 0, 0), (0, 0, 1),
                                            ignore_faces=10)])
        with self.assertRaises(TypeError):  # 非 int 元素
            packet_occluded(bvh, [ShadowRay((0, 0, 0), (0, 0, 1),
                                            ignore_faces={1.5})])


# ---------------------------------------------------------------------------
# 位掩码完整性：未命中 lane 不得被分支掩码误清
# ---------------------------------------------------------------------------

class TestMaskIntegrity(unittest.TestCase):
    def _cluster(self, cx, id_base):
        tris = []
        for i in range(6):
            z = 5.0 + i
            tris.append(Triangle.stationary(
                (cx - 1, -1, z), (cx + 1, -1, z), (cx, 1, z), face_id=id_base + i))
        return tris

    def test_unhit_lanes_survive_sibling_subtrees(self):
        # 左右两簇分属不同子树。B 在右簇最早命中并退出全部后续节点；
        # A 需走遍左簇各叶证明无遮挡、C 在左簇命中——两者的 lane 不得因
        # B 的退出或分支掩码求交被误清。
        tris = self._cluster(-10, 0) + self._cluster(10, 100)
        bvh = build(tris)
        rays = [
            ShadowRay((-9.4, 0, 0), (0, 0, 1)),  # A：穿左簇盒但落在面外 → 不挡
            ShadowRay((10, 0, 0), (0, 0, 1)),    # B：右簇第一个面即命中
            ShadowRay((-10, 0, 0), (0, 0, 1)),   # C：左簇命中
        ]
        res = packet_occluded(bvh, rays)
        self.assertEqual(res.occluded, (False, True, True))
        solo = [packet_occluded(bvh, [sr]).occluded[0] for sr in rays]
        self.assertEqual(tuple(solo), res.occluded)
        for i, sr in enumerate(rays):
            self.assertEqual(
                res.occluded[i],
                reference_occluded(tris, sr.origin, sr.direction, 0.0),
                f"lane {i}")
        # A 确实进入过叶子做了三角形测试（不是被提前清掉）
        self.assertGreater(res.triangle_tests, 0)

    def test_all_lanes_exit_early_stops_scheduling(self):
        # 全部 lane 在第一个叶命中后，其余节点不再被调度
        tri = Triangle.stationary((-2, -2, 5), (2, -2, 5), (0, 2, 5))
        filler = [Triangle.stationary(
            (10 + i, 10, 10), (11 + i, 10, 10), (10 + i, 11, 10)) for i in range(8)]
        bvh = build([tri] + filler)
        rays = [ShadowRay((0, 0, 0), (0, 0, 1)) for _ in range(16)]
        res = packet_occluded(bvh, rays)
        self.assertEqual(res.occluded, (True,) * 16)
        solo_nodes = sum(packet_occluded(bvh, [sr]).nodes_scheduled for sr in rays)
        self.assertLess(res.nodes_scheduled, solo_nodes)


# ---------------------------------------------------------------------------
# 运动场景：每条射线自己的 time
# ---------------------------------------------------------------------------

class TestPacketMotionAndWindows(unittest.TestCase):
    def test_varied_times_per_ray(self):
        # t=0 中心 x=0；t=1 中心 x=10（z=5 平面）
        a0, b0, c0 = (-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0)
        tri = Triangle(a0, b0, c0,
                       (a0[0] + 10, a0[1], a0[2]),
                       (b0[0] + 10, b0[1], b0[2]),
                       (c0[0] + 10, c0[1], c0[2]))
        bvh = build([tri])
        rays = [ShadowRay((0, 0, 0), (0, 0, 1), time=0.0),    # τ=0 命中
                ShadowRay((0, 0, 0), (0, 0, 1), time=1.0),    # τ=1 已移走
                ShadowRay((10, 0, 0), (0, 0, 1), time=1.0),   # τ=1 命中
                ShadowRay((10, 0, 0), (0, 0, 1), time=0.0)]   # τ=0 尚未移到
        res = packet_occluded(bvh, rays)
        self.assertEqual(res.occluded, (True, False, True, False))
        for i, sr in enumerate(rays):
            self.assertEqual(res.occluded[i],
                             reference_occluded([tri], sr.origin, sr.direction,
                                                sr.time), f"lane {i}")

    def test_zero_direction_components(self):
        tris = [Triangle.stationary((-1, -1, 5), (1, -1, 5), (0, 1, 5)),   # z=5
                Triangle.stationary((5, -1, -1), (5, 1, -1), (5, 0, 1))]   # x=5
        bvh = build(tris)
        rays = [ShadowRay((0, 0, 0), (0, 0, 1)),   # 命中 z=5
                ShadowRay((0, 0, 0), (0, 0, -1)),  # 反向 → 不挡
                ShadowRay((0, 0, 0), (1, 0, 0)),   # 命中 x=5
                ShadowRay((0, 2, 0), (0, 0, 1)),   # y=2 在面外 → 不挡
                ShadowRay((9, 0, 0), (0, 1, 0))]   # 与两墙平行 → 不挡
        res = packet_occluded(bvh, rays)
        self.assertEqual(res.occluded, (True, False, True, False, False))
        for i, sr in enumerate(rays):
            self.assertEqual(res.occluded[i],
                             reference_occluded(tris, sr.origin, sr.direction, 0.0),
                             f"lane {i}")


# ---------------------------------------------------------------------------
# 与独立参考的大规模比对
# ---------------------------------------------------------------------------

class TestPacketOracleFuzz(unittest.TestCase):
    def test_fuzz_against_independent_reference(self):
        rng = random.Random(20260926)
        for scene in range(4):
            n = (1, 9, 40, 120)[scene]
            base = [_rand_tri(rng) for _ in range(n)]
            ids = list(range(500, 500 + n))
            rng.shuffle(ids)
            tris = [Triangle(t.a0, t.b0, t.c0, t.a1, t.b1, t.c1, face_id=fid)
                    for t, fid in zip(base, ids)]
            bvh = build(tris)
            for _ in range(12):
                k = rng.randint(1, 64)
                rays = []
                for _ in range(k):
                    tau = rng.choice((0.0, 0.3, 0.6, 1.0))
                    if rng.random() < 0.6:
                        o, d = _aimed_ray(rng, tris, tau)
                    else:
                        o, d = _wild_ray(rng)
                    t0 = rng.uniform(0, 2)
                    t1 = math.inf if rng.random() < 0.4 else t0 + rng.uniform(0.1, 40)
                    ig = None
                    if rng.random() < 0.3 and n:
                        ig = frozenset(rng.sample(ids, rng.randint(1, min(4, n))))
                    rays.append(ShadowRay(o, d, time=tau, t_min=t0, t_max=t1,
                                          ignore_faces=ig))
                res = packet_occluded(bvh, rays)
                self.assertEqual(len(res.occluded), k)
                for i, sr in enumerate(rays):
                    exp = reference_occluded(tris, sr.origin, sr.direction,
                                             sr.time, sr.t_min, sr.t_max,
                                             sr.ignore_faces)
                    self.assertEqual(res.occluded[i], exp,
                                     f"scene {scene} lane {i}")

    def test_coherent_bundle_and_scattered_packet(self):
        # 同向成束 + 完全分散混合在一批
        rng = random.Random(4242)
        tris = _wall_scene()
        bvh = build(tris)
        rays = []
        for _ in range(40):  # 成束：同向 + 小抖动，射向 +z 墙
            o = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 0.0)
            d = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 1.0)
            rays.append(ShadowRay(o, d, time=0.0))
        for _ in range(24):  # 完全分散
            o, d = _wild_ray(rng)
            rays.append(ShadowRay(o, d, time=rng.choice((0.0, 0.5, 1.0)),
                                  t_max=rng.uniform(5, 60)))
        res = packet_occluded(bvh, rays)
        for i, sr in enumerate(rays):
            exp = reference_occluded(tris, sr.origin, sr.direction, sr.time,
                                     sr.t_min, sr.t_max)
            self.assertEqual(res.occluded[i], exp, f"lane {i}")


# ---------------------------------------------------------------------------
# 精确有理数（Fraction）独立参考：零容差逐面核对
# ---------------------------------------------------------------------------

class TestExactRationalOracle(unittest.TestCase):
    def _integer_scene(self):
        tris = [
            Triangle.stationary((0, 0, 10), (4, 0, 10), (0, 4, 10)),    # z=10 下半
            Triangle.stationary((4, 4, 10), (4, 0, 10), (0, 4, 10)),    # z=10 上半
            Triangle.stationary((0, 0, 20), (4, 0, 20), (0, 4, 20)),    # z=20
            Triangle.stationary((4, 4, 20), (4, 0, 20), (0, 4, 20)),
            Triangle.stationary((5, -2, -2), (5, 2, -2), (5, 0, 2)),    # x=5
            Triangle.stationary((-6, -2, 3), (-6, 2, 3), (-6, 0, 7)),   # x=-6
            Triangle((8, 0, 30), (12, 0, 30), (10, 4, 30),
                     (18, 0, 30), (22, 0, 30), (20, 4, 30)),            # 运动 +10x
            Triangle.stationary((0, 8, 2), (0, 8, 6), (0, 12, 4)),      # x=0 竖片
        ]
        ids = [10, 20, 30, 40, 50, 60, 70, 80]
        return [Triangle(t.a0, t.b0, t.c0, t.a1, t.b1, t.c1, face_id=f)
                for t, f in zip(tris, ids)]

    def test_exact_oracle_cross_check(self):
        tris = self._integer_scene()
        bvh = build(tris)
        packet = [
            ((2, 1, 0), (0, 0, 1), 0, 0, math.inf, None),            # z=10 挡
            ((2, 1, 0), (0, 0, 1), 0, 0, 10, None),                  # t==t_max → 不挡
            ((2, 1, 0), (0, 0, 1), 0, 10, 21, None),                 # t==t_min → 挡
            ((2, 1, 0), (0, 0, 1), 0, 11, 21, None),                 # 只余 z=20 → 挡
            ((2, 1, 0), (0, 0, 1), 0, 11, 20, None),                 # t=20==t_max → 不挡
            ((2, 1, 0), (0, 0, 1), 0, 0, math.inf, (10, 20)),        # 忽略近墙→远墙挡
            ((2, 1, 0), (0, 0, 1), 0, 0, math.inf, (10, 20, 30, 40)),  # 全忽略→不挡
            ((0, 0, 0), (1, 0, 0), 0, 0, math.inf, None),            # x=5 挡
            ((0, 0, 5), (-1, 0, 0), 0, 0, math.inf, None),           # x=-6 挡
            ((3, 10, 4), (-1, 0, 0), 0, 0, math.inf, None),          # x=0 竖片挡
            ((10, 2, 0), (0, 0, 1), 0, 0, math.inf, None),           # τ=0 运动片挡
            ((10, 2, 0), (0, 0, 1), 1, 0, math.inf, None),           # τ=1 已移走→不挡
            ((20, 2, 0), (0, 0, 1), 1, 0, math.inf, None),           # τ=1 挡
            ((3, 3, 0), (0, 0, -1), 0, 0, math.inf, None),           # 背向空处→不挡
        ]
        shadow = [ShadowRay(o, d, time=tau, t_min=t0, t_max=t1, ignore_faces=ig)
                  for (o, d, tau, t0, t1, ig) in packet]
        got = packet_occluded(bvh, shadow)
        exact = reference_packet_occluded_exact(tris, packet)
        self.assertEqual(got.occluded, exact)
        # 浮点独立参考亦应一致
        for i, sr in enumerate(shadow):
            self.assertEqual(
                got.occluded[i],
                reference_occluded(tris, sr.origin, sr.direction, sr.time,
                                   sr.t_min, sr.t_max, sr.ignore_faces),
                f"lane {i}")


# ---------------------------------------------------------------------------
# 共享调度计数
# ---------------------------------------------------------------------------

class TestPacketCounts(unittest.TestCase):
    def test_all_miss_root_counts(self):
        tri = Triangle.stationary((-1, -1, 5), (1, -1, 5), (0, 1, 5))
        bvh = build([tri])
        rays = [ShadowRay((100, 100, 100), (0, 0, 1)) for _ in range(8)]
        res = packet_occluded(bvh, rays)
        self.assertEqual(res.occluded, (False,) * 8)
        self.assertEqual(res.nodes_scheduled, 1)  # 只调度根
        self.assertEqual(res.box_tests, 8)        # 每条各测一次根盒
        self.assertEqual(res.triangle_tests, 0)

    def test_counts_are_separate_fields(self):
        rng = random.Random(5)
        tris = [_rand_tri(rng) for _ in range(40)]
        bvh = build(tris)
        rays = [ShadowRay(tuple(rng.uniform(-8, 8) for _ in range(3)),
                          tuple(rng.uniform(-1, 1) for _ in range(3)),
                          time=rng.choice((0.0, 0.5, 1.0)))
                for _ in range(24)]
        res = packet_occluded(bvh, rays)
        self.assertGreaterEqual(res.nodes_scheduled, 1)
        self.assertGreaterEqual(res.box_tests, res.nodes_scheduled)
        self.assertGreaterEqual(res.triangle_tests, 0)
        self.assertLessEqual(res.triangle_tests, 24 * 40)
        self.assertEqual(res.rejection, (None,) * 24)

    def test_shared_scheduling_less_than_per_ray(self):
        # 同向成束：整批共享调度的节点数必须严格少于逐条遍历之和；
        # 盒/三角形测试总数与逐条完全一致（同一套谓词、同一窗口）。
        rng = random.Random(20260925)
        tris = _wall_scene()
        bvh = build(tris)
        rays = []
        for _ in range(32):
            o = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 0.0)
            d = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 1.0)
            rays.append(ShadowRay(o, d, time=0.0))
        res = packet_occluded(bvh, rays)
        solo = [packet_occluded(bvh, [sr]) for sr in rays]
        sum_nodes = sum(s.nodes_scheduled for s in solo)
        self.assertLess(res.nodes_scheduled, sum_nodes)
        self.assertEqual(res.box_tests, sum(s.box_tests for s in solo))
        self.assertEqual(res.triangle_tests, sum(s.triangle_tests for s in solo))
        self.assertEqual(res.occluded, tuple(s.occluded[0] for s in solo))
        for i, sr in enumerate(rays):
            self.assertEqual(res.occluded[i],
                             reference_occluded(tris, sr.origin, sr.direction, 0.0),
                             f"lane {i}")


# ---------------------------------------------------------------------------
# 整批拒绝（全有或全无）
# ---------------------------------------------------------------------------

class TestBatchRejection(unittest.TestCase):
    def setUp(self):
        self.tri = Triangle.stationary((-1, -1, 5), (1, -1, 5), (0, 1, 5))
        self.bvh = build([self.tri])
        self.good = ShadowRay((0, 0, 0), (0, 0, 1))

    def test_any_invalid_ray_rejects_whole_batch(self):
        bads = [
            ShadowRay((0, 0, 0), (0, 0, 0)),                     # 零方向
            ShadowRay((math.nan, 0, 0), (0, 0, 1)),              # NaN 原点
            ShadowRay((0, 0, 0), (0, 0, math.inf)),              # Infinity 方向
            ShadowRay((0, 0, 0), (0, 0, 1), time=1.5),           # time 越界
            ShadowRay((0, 0, 0), (0, 0, 1), time=True),          # bool time
            ShadowRay((0, 0, 0), (0, 0, 1), t_min=6, t_max=5),   # t_min>t_max
            ShadowRay((0, 0, 0), (0, 0, 1), t_max=-math.inf),    # -inf t_max
            ShadowRay((0, 0, 0), (0, 0, 1), t_max=math.nan),     # NaN t_max
            ShadowRay((0, 0, 0), (0, 0, 1), t_min=False),        # bool t_min
            "not a shadow ray",                                  # 非 ShadowRay
            Ray((0, 0, 0), (0, 0, 1)),                           # 旧 Ray 类型
        ]
        for bad in bads:
            with self.assertRaises((TypeError, ValueError),
                                   msg=f"应整批拒绝: {bad!r}"):
                packet_occluded(self.bvh, [self.good, bad, self.good])

    def test_error_message_locates_position(self):
        try:
            packet_occluded(self.bvh, [self.good, self.good,
                                       ShadowRay((0, 0, 0), (0, 0, 0))])
        except ValueError as exc:
            self.assertIn("rays[2]", str(exc))
        else:
            self.fail("应抛出 ValueError")

    def test_failed_batch_leaves_bvh_usable(self):
        before = intersect(self.bvh, Ray((0, 0, 0), (0, 0, 1)), 0.0)
        with self.assertRaises(ValueError):
            packet_occluded(self.bvh, [ShadowRay((0, 0, 0), (0, 0, 0))])
        ok = packet_occluded(self.bvh, [self.good])
        self.assertEqual(ok.occluded, (True,))
        after = intersect(self.bvh, Ray((0, 0, 0), (0, 0, 1)), 0.0)
        self.assertEqual((before.hit, before.t), (after.hit, after.t))

    def test_packet_size_limits(self):
        with self.assertRaises(ValueError):
            packet_occluded(self.bvh, [])
        with self.assertRaises(ValueError):
            packet_occluded(self.bvh, [self.good] * (mb.MAX_PACKET + 1))
        self.assertEqual(
            len(packet_occluded(self.bvh, [self.good] * mb.MAX_PACKET).occluded),
            mb.MAX_PACKET)
        self.assertEqual(len(packet_occluded(self.bvh, [self.good]).occluded), 1)

    def test_reject_wrong_target_type(self):
        with self.assertRaises(TypeError):
            packet_occluded("bvh", [self.good])


if __name__ == "__main__":
    unittest.main(verbosity=2)
