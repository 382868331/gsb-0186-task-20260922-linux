"""实例场景（多实例共享一棵网格 BVH）的必要测试。

期望值来源：tests/reference.py 的 reference_intersect_scene —— 它把每个
实例的三角形在 τ 时刻**显式变换到世界坐标**后逐面求交，不做任何实例/
场景树遍历，也不 import motion_bvh 的算法。
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
    Triangle, Ray, build, intersect,
    Instance, build_scene, intersect_scene,
)
from reference import reference_intersect_scene


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _rand_tri(rng, motion_scale=1.5):
    def pt():
        return (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3))

    a0, b0, c0 = pt(), pt(), pt()
    a1 = tuple(a0[k] + rng.uniform(-motion_scale, motion_scale) for k in range(3))
    b1 = tuple(b0[k] + rng.uniform(-motion_scale, motion_scale) for k in range(3))
    c1 = tuple(c0[k] + rng.uniform(-motion_scale, motion_scale) for k in range(3))
    return Triangle(a0, b0, c0, a1, b1, c1)


def _at(tri, which, tau):
    p0 = getattr(tri, which + "0")
    p1 = getattr(tri, which + "1")
    return tuple(p0[k] + tau * (p1[k] - p0[k]) for k in range(3))


def _centroid_at(tri, tau):
    a, b, c = _at(tri, "a", tau), _at(tri, "b", tau), _at(tri, "c", tau)
    return tuple((a[k] + b[k] + c[k]) / 3.0 for k in range(3))


def _apply(M, T, p):
    return tuple(
        M[r][0] * p[0] + M[r][1] * p[1] + M[r][2] * p[2] + T[r]
        for r in range(3))


def _aimed_world_ray(rng, meshes, specs, tau):
    """瞄准某实例某三角形 τ 时刻世界质心构造一条非单位射线。"""
    tris, M, T = rng.choice(specs)
    tri = rng.choice(tris)
    target = _apply(M, T, _centroid_at(tri, tau))
    angle = rng.random() * 2 * math.pi
    radius = rng.uniform(6.0, 12.0)
    origin = (target[0] + radius * math.cos(angle),
              target[1] + radius * math.sin(angle),
              target[2] + rng.uniform(4.0, 10.0))
    d = tuple(target[k] - origin[k] for k in range(3))
    if rng.random() < 0.3:
        j = rng.randrange(3)
        d = list(d)
        d[j] += rng.uniform(-1.2, 1.2)
        d = tuple(d)
    return Ray(origin, d)


def _wild_ray(rng):
    return Ray(tuple(rng.uniform(-25, 25) for _ in range(3)),
               tuple(rng.uniform(-1.5, 1.5) for _ in range(3)))


def _assert_scene_ref_agreement(test, specs, scene, ray, tau,
                                t_min=0.0, t_max=math.inf):
    """specs: [(tris, M, T, iid), ...]（构造场景的同一组数据）。"""
    got = intersect_scene(scene, ray, tau, t_min, t_max)
    oracle_input = [(iid, tris, M, T) for tris, M, T, iid in specs]
    exp = reference_intersect_scene(oracle_input, ray, tau, t_min, t_max)
    test.assertIs(got.hit, exp["hit"],
                  f"命中判定不一致 τ={tau} ray={ray}")
    if not exp["hit"]:
        return got
    test.assertEqual(got.instance_id, exp["instance_id"], "instance_id 决胜不一致")
    test.assertEqual(got.face_id, exp["face_id"], "face_id 决胜不一致")
    scale = max(1.0, abs(exp["t"]))
    test.assertLessEqual(abs(got.t - exp["t"]), 1e-9 * scale,
                         f"t 不一致: {got.t} vs {exp['t']}")
    # 重心坐标是原局部网格的；仿射（含镜像）保持重心坐标
    test.assertLessEqual(abs(got.u - exp["u"]), 1e-9, "u 不一致（镜像也应保持）")
    test.assertLessEqual(abs(got.v - exp["v"]), 1e-9, "v 不一致（镜像也应保持）")
    for k in range(3):
        test.assertLessEqual(abs(got.point[k] - exp["point"][k]), 1e-7,
                             "世界命中点不一致")
    # 世界命中点必须在世界射线上：P = o + t·d
    for k in range(3):
        on = ray.origin[k] + got.t * ray.direction[k]
        test.assertLessEqual(
            abs(got.point[k] - on),
            1e-7 * max(1.0, abs(got.point[k]), abs(on)),
            "世界命中点不在世界射线上")
    return got


def _make_scene(specs):
    """specs: [(tris, M, T, iid), ...] -> (SceneBVH, cache)。每个不同 tris
    列表只 build 一次；相同 tris 列表共享同一 BVH 对象。"""
    cache = {}
    instances = []
    for tris, M, T, iid in specs:
        key = id(tris)
        if key not in cache:
            cache[key] = build(tris)
        instances.append(Instance(cache[key], iid, M, T))
    return build_scene(instances), cache


# ---------------------------------------------------------------------------
# 基础正确性：非单位射线 + 非均匀缩放 / 剪切 / 镜像
# ---------------------------------------------------------------------------

class TestTransforms(unittest.TestCase):
    def _mesh(self):
        # z=0 平面上的运动三角形：t=0 在原点附近，t=1 整体漂移
        a0, b0, c0 = (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)
        a1 = (a0[0] + 1.0, a0[1] + 0.5, 0.0)
        b1 = (b0[0] + 1.0, b0[1] + 0.5, 0.0)
        c1 = (c1x := c0[0] + 1.0, c0[1] + 0.5, 0.0)
        return [Triangle(a0, b0, c0, a1, b1, c1)]

    def test_nonuniform_scale_nonunit_ray(self):
        tris = self._mesh()
        M = ((2.0, 0.0, 0.0),
             (0.0, 0.5, 0.0),
             (0.0, 0.0, 1.5))
        T = (3.0, -4.0, 6.0)
        specs = [(tris, M, T, 0)]
        scene, _ = _make_scene(specs)
        # 明确的非单位方向（长度 ~2.7）
        ray = Ray((3.1, -3.8, 0.0), (0.13, -0.07, 2.7))
        hits = 0
        for tau in (0.0, 0.37, 1.0):
            got = _assert_scene_ref_agreement(self, specs, scene, ray, tau)
            hits += int(got.hit)
        # τ=0 与 τ=0.37 几何上被该射线击中（τ=1 漂移后允许未命中）
        self.assertGreaterEqual(hits, 2)

    def test_shear_transform(self):
        tris = self._mesh()
        M = ((1.0, 0.7, 0.0),
             (0.0, 1.0, -0.4),
             (0.2, 0.0, 1.0))
        T = (0.0, 0.0, 4.0)
        specs = [(tris, M, T, 7)]
        scene, _ = _make_scene(specs)
        rng = random.Random(101)
        for tau in (0.0, 0.5, 1.0):
            for _ in range(8):
                ray = _aimed_world_ray(rng, None, [(tris, M, T)], tau)
                _assert_scene_ref_agreement(self, specs, scene, ray, tau)

    def test_negative_determinant_mirror(self):
        tris = self._mesh()
        # 镜像 x，并叠加非均匀缩放；行列式 = -2*1*1.5 = -3
        M = ((-2.0, 0.0, 0.0),
             (0.0, 1.0, 0.0),
             (0.0, 0.0, 1.5))
        T = (1.0, 2.0, 5.0)
        det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
               - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
               + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
        self.assertLess(det, 0.0)
        specs = [(tris, M, T, 3)]
        scene, _ = _make_scene(specs)
        rng = random.Random(202)
        hits = 0
        for tau in (0.0, 0.5, 1.0):
            for _ in range(10):
                ray = _aimed_world_ray(rng, None, [(tris, M, T)], tau)
                got = _assert_scene_ref_agreement(self, specs, scene, ray, tau)
                if got.hit:
                    hits += 1
                    self.assertEqual(got.instance_id, 3)
                    # 镜像下 u/v 仍是原局部重心坐标
                    self.assertGreaterEqual(got.u, -1e-9)
                    self.assertGreaterEqual(got.v, -1e-9)
                    self.assertLessEqual(got.u + got.v, 1.0 + 1e-9)
        self.assertGreater(hits, 10)

    def test_direction_not_normalized_same_t(self):
        """方向不归一化：局部与世界射线参数 t 相同；方向放大 k 倍则 t/k。"""
        tris = self._mesh()
        M = ((1.5, 0.3, 0.0), (0.0, 0.8, 0.0), (0.0, 0.0, 2.0))
        T = (0.0, 0.0, 5.0)
        specs = [(tris, M, T, 0)]
        scene, _ = _make_scene(specs)
        # 世界三角形包含世界点 (0,0,5)（局部原点为其内点）。
        # 方向放大 3 倍 ⇒ t 缩小为 1/3，命中点不变。
        r1 = intersect_scene(scene, Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0.0)
        r2 = intersect_scene(scene, Ray((0.0, 0.0, 0.0), (0.0, 0.0, 3.0)), 0.0)
        self.assertTrue(r1.hit and r2.hit)
        self.assertAlmostEqual(r1.t, r2.t * 3.0, places=9)
        # 命中点必须相同
        for k in range(3):
            self.assertAlmostEqual(r1.point[k], r2.point[k], places=7)

    def test_identity_transform_matches_baseline(self):
        tris = self._mesh()
        bvh = build(tris)
        scene = build_scene([Instance(bvh, 0, IDENT, (0.0, 0.0, 0.0))])
        ray = Ray((0.2, -0.3, 2.0), (0.05, 0.02, 1.0))
        for tau in (0.0, 0.5, 1.0):
            base = intersect(bvh, ray, tau)
            sc = intersect_scene(scene, ray, tau)
            self.assertEqual(base.hit, sc.hit)
            if base.hit:
                self.assertEqual(base.face_id, sc.face_id)
                self.assertAlmostEqual(base.t, sc.t, places=12)
                self.assertAlmostEqual(base.u, sc.u, places=12)
                self.assertAlmostEqual(base.v, sc.v, places=12)
                for k in range(3):
                    self.assertAlmostEqual(base.point[k], sc.point[k], places=12)


class TestEndpointTimes(unittest.TestCase):
    def test_endpoint_boxes_cover_full_motion(self):
        # 三角形 t=0 位于 x=-10 一侧、t=1 漂移到 x=+10 一侧（τ 中间各处）
        a0, b0, c0 = (-11.0, -1.0, 0.0), (-9.0, -1.0, 0.0), (-10.0, 1.0, 0.0)
        a1, b1, c1 = (9.0, -1.0, 0.0), (11.0, -1.0, 0.0), (10.0, 1.0, 0.0)
        tris = [Triangle(a0, b0, c0, a1, b1, c1)]
        T = (0.0, 0.0, 4.0)
        specs = [(tris, IDENT, T, 0)]
        scene, _ = _make_scene(specs)

        # τ=0 与 τ=1 的精确瞄准射线都必须进入实例并命中（场景盒覆盖全区间）
        for tau, tx in ((0.0, -10.0), (1.0, 10.0)):
            ray = Ray((tx, 0.0, 0.0), (0.0, 0.0, 1.0))
            got = intersect_scene(scene, ray, tau)
            self.assertTrue(got.hit, f"time={tau} 端点必须命中")
            self.assertAlmostEqual(got.t, 4.0, places=10)
            _assert_scene_ref_agreement(self, specs, scene, ray, tau)

        # τ=0.5 三角形居中
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        _assert_scene_ref_agreement(self, specs, scene, ray, 0.5)


# ---------------------------------------------------------------------------
# 同距决胜：先 instance_id 再 face_id
# ---------------------------------------------------------------------------

class TestSceneTie(unittest.TestCase):
    def _big_triangle(self):
        return [Triangle.stationary(
            (-5.0, -5.0, 0.0), (5.0, -5.0, 0.0), (0.0, 5.0, 0.0))]

    def test_two_instances_same_distance_lower_id_wins(self):
        tris = self._big_triangle()
        bvh = build(tris)
        # 两个实例完全重合放置 ⇒ 每次命中 t 位相同
        inst_hi = Instance(bvh, 5, IDENT, (0.0, 0.0, 3.0))
        inst_lo = Instance(bvh, 2, IDENT, (0.0, 0.0, 3.0))
        scene = build_scene([inst_hi, inst_lo])  # 大 id 在前
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(scene, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertEqual(res.instance_id, 2)
        self.assertAlmostEqual(res.t, 3.0, places=12)
        specs = [(tris, IDENT, (0.0, 0.0, 3.0), 5),
                 (tris, IDENT, (0.0, 0.0, 3.0), 2)]
        _assert_scene_ref_agreement(self, specs, scene, ray, 0.0)

    def test_strictly_nearer_instance_beats_tie_rule(self):
        tris = self._big_triangle()
        bvh = build(tris)
        near = Instance(bvh, 99, IDENT, (0.0, 0.0, 2.0))
        far = Instance(bvh, 1, IDENT, (0.0, 0.0, 8.0))
        scene = build_scene([near, far])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(scene, ray, 0.0)
        self.assertEqual(res.instance_id, 99)
        self.assertAlmostEqual(res.t, 2.0, places=12)

    def test_tie_then_face_id_within_instance(self):
        # 同一实例内两个共面三角形（face_id 2/9），场景层也要按 face_id 决胜
        a = Triangle.stationary(
            (-5.0, -5.0, 1.0), (5.0, -5.0, 1.0), (0.0, 5.0, 1.0), face_id=9)
        b = Triangle.stationary(
            (-5.0, 5.0, 1.0), (5.0, 5.0, 1.0), (0.0, -5.0, 1.0), face_id=2)
        bvh = build([a, b])
        scene = build_scene([Instance(bvh, 4, IDENT, (0.0, 0.0, 0.0))])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(scene, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertEqual((res.instance_id, res.face_id), (4, 2))
        specs = [([a, b], IDENT, (0.0, 0.0, 0.0), 4)]
        _assert_scene_ref_agreement(self, specs, scene, ray, 0.0)


# ---------------------------------------------------------------------------
# 共享底层树 / 不复制面
# ---------------------------------------------------------------------------

class TestSharedBVH(unittest.TestCase):
    def test_two_instances_share_one_tree(self):
        tris = [Triangle.stationary(
            (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))]
        bvh = build(tris)
        i1 = Instance(bvh, 0, IDENT, (0.0, 0.0, 3.0))
        M2 = ((0.7, 0.0, 0.0), (0.0, 0.7, 0.0), (0.0, 0.0, 0.7))
        i2 = Instance(bvh, 1, M2, (6.0, 0.0, 3.0))
        scene = build_scene([i1, i2])
        self.assertEqual(scene.instance_count, 2)
        # 自省：场景记录的 BVH 与面存储就是同一个对象
        seen_bvh = []

        def walk(node):
            if node.items is not None:
                for rec in node.items:
                    seen_bvh.append(rec[1])
            else:
                walk(node.left)
                walk(node.right)

        walk(scene._root)
        self.assertEqual(len(seen_bvh), 2)
        self.assertIs(seen_bvh[0], bvh)
        self.assertIs(seen_bvh[1], bvh)
        self.assertIs(scene._shared_bvh, bvh)

        # 两个实例各自能被同一条/不同射线命中
        r1 = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        r2 = Ray((6.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res1 = intersect_scene(scene, r1, 0.0)
        res2 = intersect_scene(scene, r2, 0.0)
        self.assertTrue(res1.hit and res2.hit)
        self.assertEqual(res1.instance_id, 0)
        self.assertEqual(res2.instance_id, 1)

    def test_distinct_meshes_also_supported(self):
        rng = random.Random(11)
        m1 = [_rand_tri(rng) for _ in range(12)]
        m2 = [_rand_tri(rng) for _ in range(8)]
        b1, b2 = build(m1), build(m2)
        scene = build_scene([
            Instance(b1, 0, IDENT, (-8.0, 0.0, 0.0)),
            Instance(b2, 1, IDENT, (8.0, 0.0, 0.0))])
        self.assertIsNone(scene._shared_bvh)  # 自省字段：非全共享
        specs = [(m1, IDENT, (-8.0, 0.0, 0.0), 0),
                 (m2, IDENT, (8.0, 0.0, 0.0), 1)]
        for tau in (0.0, 0.5, 1.0):
            for _ in range(15):
                ray = _aimed_world_ray(
                    rng, None,
                    [(m1, IDENT, (-8.0, 0.0, 0.0)),
                     (m2, IDENT, (8.0, 0.0, 0.0))], tau)
                _assert_scene_ref_agreement(self, specs, scene, ray, tau)


# ---------------------------------------------------------------------------
# 计 数
# ---------------------------------------------------------------------------

class TestSceneCounts(unittest.TestCase):
    def test_three_kinds_of_counts_distinguished(self):
        tris = []
        for i in range(6):
            tris.append(Triangle.stationary(
                (i * 0.5, 0.0, 0.0), (i * 0.5 + 0.4, 0.0, 0.0),
                (i * 0.5 + 0.2, 1.0, 0.0)))
        bvh = build(tris)

        def cluster(specs):
            return [Instance(bvh, iid,
                             ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                             (0.0, 0.0, z))
                    for iid, z in specs]

        # 4 个实例在 z=40 一簇，4 个在 z=-40 一簇（不同 instance_id）
        instances = (cluster([(i, 40.0 + i) for i in range(4)])
                     + cluster([(i + 4, -40.0 - i) for i in range(4)]))
        scene = build_scene(instances)

        # 射线明显偏离所有几何：只测场景盒，底层盒与三角形都为 0
        miss = intersect_scene(scene, Ray((1000.0, 1000.0, 0.0),
                                          (0.0, 0.0, 1.0)), 0.0)
        self.assertFalse(miss.hit)
        self.assertGreaterEqual(miss.scene_boxes_tested, 1)
        self.assertEqual(miss.underlying_boxes_tested, 0)
        self.assertEqual(miss.triangles_tested, 0)

        # 射向 +z 簇某个实例：-z 簇整叶被场景层剪掉，底层盒只来自 +z 簇
        hit = intersect_scene(scene, Ray((1.2, 0.3, 0.0),
                                         (0.0, 0.0, 1.0)), 0.0)
        self.assertTrue(hit.hit)
        self.assertIn(hit.instance_id, (0, 1, 2, 3))
        # 每个被进入的实例底层树至少测根节点；未进入的实例 0 次
        self.assertLessEqual(hit.underlying_boxes_tested, 4 * 7)
        self.assertLess(hit.triangles_tested, 8 * len(tris))
        self.assertGreaterEqual(hit.scene_boxes_tested, 1)


# ---------------------------------------------------------------------------
# 空场景 / 空底层网格
# ---------------------------------------------------------------------------

class TestEmptyScene(unittest.TestCase):
    def test_empty_scene(self):
        scene = build_scene([])
        self.assertEqual(scene.instance_count, 0)
        res = intersect_scene(scene, Ray((0, 0, 0), (0, 0, 1)), 0.0)
        self.assertFalse(res.hit)
        self.assertEqual(res.scene_boxes_tested, 0)
        self.assertEqual(res.underlying_boxes_tested, 0)
        self.assertEqual(res.triangles_tested, 0)
        self.assertIsNone(res.instance_id)
        self.assertTrue(math.isnan(res.t))

    def test_instance_referencing_empty_bvh(self):
        empty = build([])
        good = build([Triangle.stationary(
            (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))])
        scene = build_scene([
            Instance(empty, 0, IDENT, (0.0, 0.0, 5.0)),
            Instance(good, 1, IDENT, (0.0, 0.0, 2.0))])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(scene, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertEqual(res.instance_id, 1)  # 空实例永不命中
        specs = [([], IDENT, (0.0, 0.0, 5.0), 0),
                 ([Triangle.stationary(
                     (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))],
                  IDENT, (0.0, 0.0, 2.0), 1)]
        _assert_scene_ref_agreement(self, specs, scene, ray, 0.0)


# ---------------------------------------------------------------------------
# 随机差分：共享网格 + 各种可逆变换
# ---------------------------------------------------------------------------

class TestSceneFuzz(unittest.TestCase):
    def test_random_instances_against_worldbrute_reference(self):
        rng = random.Random(20260925)
        meshes = [[_rand_tri(rng) for _ in range(n)]
                  for n in (10, 25, 60)]
        bvhs = [build(m) for m in meshes]
        specs = []
        next_id = 0
        for _ in range(14):
            mi = rng.randrange(len(meshes))
            # 随机可逆线性部分：旋转(用 cos/sin 拼) + 非均匀缩放 + 剪切，
            # 约 1/3 概率加镜像
            ang = rng.uniform(0, 2 * math.pi)
            c, s = math.cos(ang), math.sin(ang)
            rot = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
            sx = rng.choice((-1.0, 1.0)) if rng.random() < 0.34 else 1.0
            scale = ((sx * rng.uniform(0.3, 2.5), 0.0, 0.0),
                     (0.0, rng.uniform(0.3, 2.5), 0.0),
                     (0.0, 0.0, rng.uniform(0.3, 2.5)))
            sh = ((1.0, rng.uniform(-0.6, 0.6), rng.uniform(-0.4, 0.4)),
                  (0.0, 1.0, rng.uniform(-0.4, 0.4)),
                  (0.0, 0.0, 1.0))
            M = _mat_mul(_mat_mul(rot, scale), sh)
            T = (rng.uniform(-12, 12), rng.uniform(-12, 12),
                 rng.uniform(-12, 12))
            specs.append((meshes[mi], M, T, next_id))
            next_id += 1

        instances = []
        for tris, M, T, iid in specs:
            mi = next(j for j, m in enumerate(meshes) if m is tris)
            instances.append(Instance(bvhs[mi], iid, M, T))
        scene = build_scene(instances)

        spec_specs = specs
        rays = 0
        for _ in range(180):
            tau = rng.choice((0.0, 0.13, 0.5, 0.87, 1.0))
            if rng.random() < 0.65:
                placed = [(tris, M, T) for tris, M, T, _ in spec_specs]
                ray = _aimed_world_ray(rng, None, placed, tau)
            else:
                ray = _wild_ray(rng)
            _assert_scene_ref_agreement(self, spec_specs, scene, ray, tau)
            rays += 1
        self.assertGreater(rays, 100)


def _mat_mul(a, b):
    return tuple(tuple(sum(a[r][k] * b[k][c] for k in range(3))
                       for c in range(3)) for r in range(3))


# ---------------------------------------------------------------------------
# 校验：全有或全无
# ---------------------------------------------------------------------------

class TestSceneValidation(unittest.TestCase):
    def setUp(self):
        self.bvh = build([Triangle.stationary(
            (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))])

    def _ok(self, iid=0, M=IDENT, T=(0.0, 0.0, 1.0)):
        return Instance(self.bvh, iid, M, T)

    def test_duplicate_instance_id_rejected(self):
        with self.assertRaises(ValueError):
            build_scene([self._ok(3),
                         Instance(self.bvh, 3, IDENT, (5.0, 0.0, 1.0))])

    def test_negative_and_bool_instance_id(self):
        with self.assertRaises(ValueError):
            build_scene([self._ok(-1)])
        with self.assertRaises(TypeError):
            build_scene([Instance(self.bvh, True, IDENT)])

    def test_non_finite_matrix_values_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            M = ((bad, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            with self.assertRaises(ValueError):
                build_scene([Instance(self.bvh, 0, M)])

    def test_non_finite_translation_rejected(self):
        with self.assertRaises(ValueError):
            build_scene([Instance(self.bvh, 0, IDENT, (math.nan, 0.0, 0.0))])

    def test_singular_matrix_rejected(self):
        singulars = [
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),     # 零行
            ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (8.0, 10.0, 12.0)),   # 行3=2行2
            ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),     # 两列相关
        ]
        for M in singulars:
            with self.assertRaises(ValueError):
                build_scene([Instance(self.bvh, 0, M)])

    def test_inverse_nonfinite_rejected_without_loose_tolerance(self):
        # 元素过大：余因子乘积溢出为 inf，逆矩阵出现 inf/NaN，必须拒绝；
        # 而 det 很小但严格非零的合法矩阵必须照常接受（不随意收紧容差）。
        huge = ((1e200, 0.0, 0.0),
                (0.0, 1e200, 0.0),
                (0.0, 0.0, -1e200))
        with self.assertRaises(ValueError):
            build_scene([Instance(self.bvh, 0, huge)])

        tiny = ((1e-150, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0))
        scene = build_scene([Instance(self.bvh, 0, tiny, (0.0, 0.0, 2.0))])
        res = intersect_scene(scene, Ray((0.0, 0.0, 0.0),
                                         (0.0, 0.0, 1.0)), 0.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 2.0, places=10)

    def test_matrix_shape_and_types(self):
        # Instance 本身是不带校验的数据载体，形状错误在 build_scene 时定位拒绝
        with self.assertRaises(TypeError):
            build_scene([Instance(self.bvh, 0, ((1, 0, 0), (0, 1, 0)))])  # 2 行
        with self.assertRaises(TypeError):
            build_scene([Instance(self.bvh, 0, ((1, 0), (0, 1), (0, 0)))])
        with self.assertRaises(TypeError):
            build_scene([Instance(self.bvh, 0, "not a matrix")])
        # bool 不充当数值
        with self.assertRaises(TypeError):
            build_scene([Instance(
                self.bvh, 0,
                ((True, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))])
        # 错误行类型
        with self.assertRaises(TypeError):
            build_scene([Instance(self.bvh, 0, [1, 2, 3])])

    def test_bvh_must_be_bvh_instance(self):
        with self.assertRaises(TypeError):
            build_scene([Instance("bvh", 0, IDENT)])
        with self.assertRaises(TypeError):
            build_scene([Instance(object(), 0, IDENT)])

    def test_instance_element_type(self):
        with self.assertRaises(TypeError):
            build_scene([(self.bvh, 0, IDENT)])

    def test_failed_scene_build_is_all_or_nothing(self):
        # 先建一个好场景并记录结果
        good_scene = build_scene([
            Instance(self.bvh, 0, IDENT, (0.0, 0.0, 2.0))])
        before = intersect_scene(
            good_scene, Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0.0)
        self.assertTrue(before.hit)

        # 非法批次（重复 id 在第 3 个实例）必须整体失败
        bad = [self._ok(0, T=(0.0, 0.0, 1.0)),
               self._ok(1, T=(3.0, 0.0, 1.0)),
               self._ok(0, T=(6.0, 0.0, 1.0))]
        with self.assertRaises(ValueError):
            build_scene(bad)

        # 旧场景/底层 BVH 行为不变
        after = intersect_scene(
            good_scene, Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0.0)
        self.assertEqual(after.instance_id, before.instance_id)
        self.assertEqual(after.t, before.t)
        # 底层 BVH 仍可独立查询（旧接口未受影响）
        base = intersect(self.bvh, Ray((0.0, 0.0, -2.0), (0.0, 0.0, 1.0)), 0.0)
        self.assertTrue(base.hit)

    def test_scene_bvh_not_directly_constructible(self):
        with self.assertRaises(TypeError):
            mb.SceneBVH()

    def test_scene_query_validation(self):
        scene = build_scene([self._ok()])
        with self.assertRaises(ValueError):  # 零方向
            intersect_scene(scene, Ray((0, 0, 0), (0, 0, 0)), 0.0)
        with self.assertRaises(ValueError):  # time 越界
            intersect_scene(scene, Ray((0, 0, 0), (0, 0, 1)), 1.5)
        with self.assertRaises(TypeError):  # 非 Ray
            intersect_scene(scene, ((0, 0, 0), (0, 0, 1)), 0.0)
        with self.assertRaises(TypeError):  # 非 SceneBVH
            intersect_scene("scene", Ray((0, 0, 0), (0, 0, 1)), 0.0)
        with self.assertRaises(ValueError):  # t_min > t_max
            intersect_scene(scene, Ray((0, 0, 0), (0, 0, 1)), 0.0,
                            t_min=5, t_max=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
