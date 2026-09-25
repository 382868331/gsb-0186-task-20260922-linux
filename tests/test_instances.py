"""motion_bvh 实例/场景层必要测试。

期望值来源：tests/reference.py 中 reference_scene_intersect —— 把三角形
显式变换到世界坐标后逐面暴力求交，不借用被测模块的实例遍历。
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
    Triangle, Ray, build, Instance, build_scene, intersect_scene,
)
from reference import reference_scene_intersect


# ---------------------------------------------------------------------------
# 确定性数据生成
# ---------------------------------------------------------------------------

def _rand_tri(rng, motion_scale=1.5):
    def pt():
        return (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3))

    a0, b0, c0 = pt(), pt(), pt()
    a1 = tuple(a0[k] + rng.uniform(-motion_scale, motion_scale) for k in range(3))
    b1 = tuple(b0[k] + rng.uniform(-motion_scale, motion_scale) for k in range(3))
    c1 = tuple(c0[k] + rng.uniform(-motion_scale, motion_scale) for k in range(3))
    return Triangle(a0, b0, c0, a1, b1, c1)


def _rotz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _matmul(A, B):
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _rand_transform(rng, allow_mirror=True):
    """旋转 @ 非均匀缩放 @ 剪切，偶尔镜像；平移随机。"""
    ang = rng.uniform(0, 2 * math.pi)
    rot = _rotz(ang)
    sx = rng.uniform(0.4, 2.5) * (-1.0 if allow_mirror and rng.random() < 0.25 else 1.0)
    sy = rng.uniform(0.4, 2.5)
    sz = rng.uniform(0.4, 2.5)
    scale = ((sx, 0.0, 0.0), (0.0, sy, 0.0), (0.0, 0.0, sz))
    sh = rng.uniform(-0.4, 0.4)
    shear = ((1.0, sh, 0.0), (0.0, 1.0, sh), (0.0, 0.0, 1.0))
    m = _matmul(rot, _matmul(scale, shear))
    t = (rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(-15, 15))
    return m, t


def _world_centroid(tri, mat, trans, tau):
    def at(p0, p1):
        return tuple(p0[k] + tau * (p1[k] - p0[k]) for k in range(3))

    a, b, c = at(tri.a0, tri.a1), at(tri.b0, tri.b1), at(tri.c0, tri.c1)
    lc = tuple((a[k] + b[k] + c[k]) / 3.0 for k in range(3))
    wc = tuple(sum(mat[i][k] * lc[k] for k in range(3)) + trans[i]
               for i in range(3))
    return wc


def _assert_scene_ref_agreement(test, scene, ref_instances, ray, tau,
                                t_min=0.0, t_max=math.inf,
                                t_tol=1e-9, uv_tol=1e-9):
    got = intersect_scene(scene, ray, tau, t_min, t_max)
    exp = reference_scene_intersect(ref_instances, ray, tau, t_min, t_max)
    test.assertIs(got.hit, exp["hit"],
                  f"命中判定不一致 τ={tau} ray={ray}")
    if not exp["hit"]:
        return got
    test.assertEqual(got.instance_id, exp["instance_id"], "instance_id 不一致")
    test.assertEqual(got.face_id, exp["face_id"], "face_id / 同距决胜不一致")
    scale = max(1.0, abs(exp["t"]))
    test.assertLessEqual(abs(got.t - exp["t"]), t_tol * scale,
                         f"t 不一致: {got.t} vs {exp['t']}")
    test.assertLessEqual(abs(got.u - exp["u"]), uv_tol, "u 不一致")
    test.assertLessEqual(abs(got.v - exp["v"]), uv_tol, "v 不一致")
    for k in range(3):
        test.assertLessEqual(abs(got.point[k] - exp["point"][k]), 1e-7,
                             f"世界命中点不一致: {got.point} vs {exp['point']}")
    return got


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

class TestSharedBVH(unittest.TestCase):
    def test_two_instances_share_one_bvh(self):
        tri = Triangle.stationary(
            (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))
        bvh = build([tri])
        ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        inst_a = Instance(bvh, 10, ident, (0.0, 0.0, 5.0))
        inst_b = Instance(bvh, 20, ident, (0.0, 0.0, 10.0))
        scene = build_scene([inst_a, inst_b])
        # 同一棵底层树被两个实例引用，未复制重建
        self.assertIs(scene._recs[0].bvh, bvh)
        self.assertIs(scene._recs[1].bvh, bvh)
        self.assertEqual(scene.instance_count, 2)
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(scene, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertEqual(res.instance_id, 10)  # 更近的实例
        self.assertAlmostEqual(res.t, 5.0, places=10)
        self.assertAlmostEqual(res.point[2], 5.0, places=10)
        ref = [(10, ident, (0.0, 0.0, 5.0), [tri]),
               (20, ident, (0.0, 0.0, 10.0), [tri])]
        _assert_scene_ref_agreement(self, scene, ref, ray, 0.0)

    def test_old_api_still_works(self):
        tri = Triangle.stationary(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        bvh = build([tri])
        res = mb.intersect(bvh, Ray((0.2, 0.3, -1.0), (0.0, 0.0, 1.0)), 0.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 1.0, places=12)


class TestTransformKinds(unittest.TestCase):
    """非均匀缩放 + 非单位射线、镜像、剪切、旋转，均与世界坐标参考比对。"""

    def setUp(self):
        rng = random.Random(555)
        self.tris = [_rand_tri(rng) for _ in range(12)]
        self.bvh = build(self.tris)

    def _one_instance_scene(self, iid, mat, trans):
        scene = build_scene([Instance(self.bvh, iid, mat, trans)])
        ref = [(iid, mat, trans, self.tris)]
        return scene, ref

    def test_nonuniform_scale_with_nonunit_ray(self):
        mat = ((2.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 3.0))
        trans = (4.0, -2.0, 7.0)
        scene, ref = self._one_instance_scene(1, mat, trans)
        rng = random.Random(41)
        checked = 0
        for _ in range(60):
            tau = rng.choice((0.0, 0.3, 0.7, 1.0))
            tgt = _world_centroid(rng.choice(self.tris), mat, trans, tau)
            origin = (tgt[0] + rng.uniform(-12, 12),
                      tgt[1] + rng.uniform(-12, 12),
                      tgt[2] + rng.uniform(-12, 12))
            # 非单位方向：长度 0.3~7，t 以该方向为单位
            d = tuple((tgt[k] - origin[k]) * rng.uniform(0.3, 7.0)
                      for k in range(3))
            ray = Ray(origin, d)
            got = _assert_scene_ref_agreement(self, scene, ref, ray, tau)
            if got.hit:
                checked += 1
                # 世界命中点与 origin + t*direction 一致（t 未因局部变换走样）
                for k in range(3):
                    self.assertAlmostEqual(
                        got.point[k], origin[k] + got.t * d[k], places=6)
        self.assertGreater(checked, 20)

    def test_negative_determinant_mirror(self):
        # x 镜像 + 缩放：行列式为负
        mat = ((-1.5, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.8))
        trans = (1.0, 2.0, 3.0)
        det = -1.5 * 1.0 * 0.8
        self.assertLess(det, 0.0)
        scene, ref = self._one_instance_scene(2, mat, trans)
        rng = random.Random(42)
        hits = 0
        for _ in range(50):
            tau = rng.choice((0.0, 0.5, 1.0))
            tgt = _world_centroid(rng.choice(self.tris), mat, trans, tau)
            origin = (tgt[0] + rng.uniform(-10, 10),
                      tgt[1] + rng.uniform(-10, 10),
                      tgt[2] + rng.uniform(-10, 10))
            d = tuple(tgt[k] - origin[k] for k in range(3))
            got = _assert_scene_ref_agreement(
                self, scene, ref, Ray(origin, d), tau)
            hits += got.hit
        self.assertGreater(hits, 15)

    def test_shear_and_rotation(self):
        mat = _matmul(_rotz(0.7), ((1.0, 0.6, 0.2), (0.0, 1.0, 0.3),
                                   (0.1, 0.0, 1.0)))
        trans = (-5.0, 3.0, -1.0)
        scene, ref = self._one_instance_scene(3, mat, trans)
        rng = random.Random(43)
        for _ in range(50):
            tau = rng.choice((0.0, 0.25, 0.9, 1.0))
            tgt = _world_centroid(rng.choice(self.tris), mat, trans, tau)
            origin = (tgt[0] + rng.uniform(-10, 10),
                      tgt[1] + rng.uniform(-10, 10),
                      tgt[2] + rng.uniform(-10, 10))
            d = tuple(tgt[k] - origin[k] for k in range(3))
            _assert_scene_ref_agreement(self, scene, ref, Ray(origin, d), tau)


class TestSceneTies(unittest.TestCase):
    def test_two_instances_same_distance_lower_instance_id_wins(self):
        # 两个实例完全同变换（几何重合），同距时 instance_id 小者胜
        tri = Triangle.stationary(
            (-2.0, -2.0, 1.0), (2.0, -2.0, 1.0), (0.0, 2.0, 1.0), face_id=4)
        bvh = build([tri])
        ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        zero = (0.0, 0.0, 0.0)
        scene = build_scene([Instance(bvh, 9, ident, zero),
                             Instance(bvh, 3, ident, zero)])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(scene, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertEqual(res.instance_id, 3)
        self.assertEqual(res.face_id, 4)
        self.assertAlmostEqual(res.t, 1.0, places=12)
        ref = [(9, ident, zero, [tri]), (3, ident, zero, [tri])]
        _assert_scene_ref_agreement(self, scene, ref, ray, 0.0)

    def test_tie_within_instance_still_by_face_id(self):
        # 同一实例内两个共面重合三角形：同距先 instance（相同）再 face_id
        a = Triangle.stationary(
            (-2.0, -2.0, 1.0), (2.0, -2.0, 1.0), (0.0, 2.0, 1.0), face_id=8)
        b = Triangle.stationary(
            (-2.0, 2.0, 1.0), (2.0, 2.0, 1.0), (0.0, -2.0, 1.0), face_id=5)
        bvh = build([a, b])
        ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        scene = build_scene([Instance(bvh, 1, ident, (0.0, 0.0, 0.0))])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(scene, ray, 0.0)
        self.assertEqual((res.instance_id, res.face_id), (1, 5))

    def test_strictly_closer_instance_beats_id_rule(self):
        tri = Triangle.stationary(
            (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))
        bvh = build([tri])
        ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        # id 小的实例更远：严格更近者必须胜
        scene = build_scene([Instance(bvh, 1, ident, (0.0, 0.0, 9.0)),
                             Instance(bvh, 2, ident, (0.0, 0.0, 4.0))])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(scene, ray, 0.0)
        self.assertEqual(res.instance_id, 2)
        self.assertAlmostEqual(res.t, 4.0, places=12)


class TestSceneEmptyAndEndpoints(unittest.TestCase):
    def test_empty_scene(self):
        scene = build_scene([])
        self.assertEqual(scene.instance_count, 0)
        res = intersect_scene(scene, Ray((0, 0, 0), (0, 0, 1)), 0.0)
        self.assertFalse(res.hit)
        self.assertEqual(res.scene_boxes_tested, 0)
        self.assertEqual(res.mesh_boxes_tested, 0)
        self.assertEqual(res.triangles_tested, 0)
        self.assertIsNone(res.instance_id)
        self.assertIsNone(res.face_id)
        self.assertTrue(math.isnan(res.t))

    def test_scene_with_empty_mesh_instance(self):
        empty_bvh = build([])
        tri = Triangle.stationary(
            (-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0))
        bvh = build([tri])
        ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        scene = build_scene([Instance(empty_bvh, 1, ident, (0.0, 0.0, 0.0)),
                             Instance(bvh, 2, ident, (0.0, 0.0, 0.0))])
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(scene, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertEqual(res.instance_id, 2)
        self.assertEqual(res.triangles_tested, 1)  # 空实例不测任何三角形

    def test_time_endpoints_through_instance(self):
        # 运动网格经实例变换后，τ=0 与 τ=1 端点精确命中对应位置
        a0, b0, c0 = (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 1.0, 1.0)
        a1 = (a0[0] + 4.0, a0[1], a0[2])
        b1 = (b0[0] + 4.0, b0[1], b0[2])
        c1 = (c0[0] + 4.0, c0[1], c0[2])
        tri = Triangle(a0, b0, c0, a1, b1, c1)
        bvh = build([tri])
        mat = ((0.5, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0))
        trans = (10.0, 0.0, 0.0)
        scene = build_scene([Instance(bvh, 7, mat, trans)])
        ref = [(7, mat, trans, [tri])]
        # τ=0：世界中心 x = 10 + 0.5*0 = 10, z = 1
        r0 = Ray((10.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res0 = _assert_scene_ref_agreement(self, scene, ref, r0, 0.0)
        self.assertTrue(res0.hit)
        self.assertAlmostEqual(res0.point[0], 10.0, places=9)
        # τ=1：局部中心 x=4，世界 x = 10 + 0.5*4 = 12
        r1 = Ray((12.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res1 = _assert_scene_ref_agreement(self, scene, ref, r1, 1.0)
        self.assertTrue(res1.hit)
        self.assertAlmostEqual(res1.point[0], 12.0, places=9)
        # τ=0 时瞄准 τ=1 的位置必须未命中（运动区间不被冻结）
        self.assertFalse(intersect_scene(scene, r1, 0.0).hit)


class TestSceneWindowAndCounts(unittest.TestCase):
    def setUp(self):
        self.tri = Triangle.stationary(
            (-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (0.0, 1.0, 5.0))
        self.bvh = build([self.tri])
        ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self.ident = ident
        self.scene = build_scene([
            Instance(self.bvh, 1, ident, (0.0, 0.0, 0.0)),
            Instance(self.bvh, 2, ident, (50.0, 0.0, 0.0)),
        ])
        self.refs = [(1, ident, (0.0, 0.0, 0.0), [self.tri]),
                     (2, ident, (50.0, 0.0, 0.0), [self.tri])]

    def test_t_min_t_max_window(self):
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertFalse(intersect_scene(self.scene, ray, 0.0, t_min=6.0).hit)
        self.assertFalse(intersect_scene(self.scene, ray, 0.0, t_max=4.0).hit)
        res = intersect_scene(self.scene, ray, 0.0, t_min=5.0, t_max=5.0)
        self.assertTrue(res.hit)
        self.assertAlmostEqual(res.t, 5.0, places=12)
        _assert_scene_ref_agreement(self, self.scene, self.refs, ray, 0.0,
                                    t_min=5.0, t_max=5.0)

    def test_counts_distinguish_three_levels(self):
        # 命中近处实例：场景盒测试 >= 2（根 + 叶），底层盒测试 >= 1，测面 1
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(self.scene, ray, 0.0)
        self.assertTrue(res.hit)
        self.assertGreaterEqual(res.scene_boxes_tested, 2)
        self.assertGreaterEqual(res.mesh_boxes_tested, 1)
        self.assertEqual(res.triangles_tested, 1)

    def test_proved_miss_counts(self):
        # 明显不命中任何实例盒：测面 0，但场景根盒一定测过
        ray = Ray((1000.0, 1000.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(self.scene, ray, 0.0)
        self.assertFalse(res.hit)
        self.assertEqual(res.triangles_tested, 0)
        self.assertEqual(res.mesh_boxes_tested, 0)
        self.assertGreaterEqual(res.scene_boxes_tested, 1)

    def test_far_instance_pruned_after_near_hit(self):
        # 命中 z=5 的近实例后，远处实例（x=50）应被场景盒剪枝
        ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        res = intersect_scene(self.scene, ray, 0.0)
        self.assertEqual(res.instance_id, 1)
        self.assertEqual(res.triangles_tested, 1)


class TestSceneFuzz(unittest.TestCase):
    def test_random_transforms_vs_reference(self):
        rng = random.Random(20260925)
        tris = [_rand_tri(rng) for _ in range(22)]
        bvh = build(tris)
        instances = []
        refs = []
        for i in range(5):
            mat, trans = _rand_transform(rng)
            iid = (i + 1) * 100
            instances.append(Instance(bvh, iid, mat, trans))
            refs.append((iid, mat, trans, tris))
        scene = build_scene(instances)
        self.assertEqual(scene.instance_count, 5)

        hits = 0
        for qi in range(300):
            tau = rng.choice((0.0, 0.13, 0.5, 0.87, 1.0))
            if rng.random() < 0.7:
                iid, mat, trans, _ = rng.choice(refs)
                tgt = _world_centroid(rng.choice(tris), mat, trans, tau)
                origin = (tgt[0] + rng.uniform(-20, 20),
                          tgt[1] + rng.uniform(-20, 20),
                          tgt[2] + rng.uniform(-20, 20))
                d = tuple((tgt[k] - origin[k]) * rng.uniform(0.2, 5.0)
                          for k in range(3))
            else:
                origin = tuple(rng.uniform(-30, 30) for _ in range(3))
                d = tuple(rng.uniform(-1, 1) for _ in range(3))
            got = _assert_scene_ref_agreement(
                self, scene, refs, Ray(origin, d), tau)
            hits += got.hit
        self.assertGreater(hits, 50)


class TestSceneValidation(unittest.TestCase):
    def setUp(self):
        self.tri = Triangle.stationary(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        self.bvh = build([self.tri])
        self.ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self.zero = (0.0, 0.0, 0.0)

    def _inst(self, iid=1, mat=None, trans=None, bvh=None):
        return Instance(bvh or self.bvh, iid,
                        mat if mat is not None else self.ident,
                        trans if trans is not None else self.zero)

    def test_reject_duplicate_instance_id(self):
        with self.assertRaises(ValueError):
            build_scene([self._inst(5), self._inst(5)])

    def test_reject_negative_and_bool_instance_id(self):
        with self.assertRaises(ValueError):
            build_scene([self._inst(-1)])
        with self.assertRaises(TypeError):
            build_scene([self._inst(True)])
        with self.assertRaises(TypeError):
            build_scene([self._inst(1.5)])

    def test_reject_non_finite_in_matrix_and_translation(self):
        for bad in (math.nan, math.inf, -math.inf):
            m = [list(r) for r in self.ident]
            m[0][0] = bad
            with self.assertRaises(ValueError):
                build_scene([self._inst(1, mat=m)])
            with self.assertRaises(ValueError):
                build_scene([self._inst(1, trans=(bad, 0.0, 0.0))])
        with self.assertRaises(TypeError):
            build_scene([self._inst(1, mat=((True, 0, 0), (0, 1, 0), (0, 0, 1)))])

    def test_reject_singular_matrix(self):
        singulars = [
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),   # 零行
            ((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (0.0, 1.0, 0.0)),   # 行相关
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),   # z 行为和
            ((1e-300, 0.0, 0.0), (0.0, 1e-300, 0.0),
             (0.0, 0.0, 1e-300)),                                   # 行列式下溢为 0
        ]
        for m in singulars:
            with self.assertRaises(ValueError, msg=f"应拒绝奇异矩阵 {m}"):
                build_scene([self._inst(1, mat=m)])

    def test_reject_non_finite_inverse(self):
        # 行列式非零（次正规数）但逆矩阵元溢出为 inf：必须拒绝，不得放行
        m = ((1e-310, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            build_scene([self._inst(1, mat=m)])

    def test_reject_wrong_types(self):
        with self.assertRaises(TypeError):
            build_scene("not instances")
        with self.assertRaises(TypeError):
            build_scene([(self.bvh, 1, self.ident, self.zero)])
        with self.assertRaises(TypeError):
            build_scene([Instance("bvh", 1, self.ident, self.zero)])
        with self.assertRaises(TypeError):
            build_scene([self._inst(1, mat=((1, 0, 0), (0, 1, 0)))])
        with self.assertRaises(TypeError):
            intersect_scene("scene", Ray((0, 0, 0), (0, 0, 1)), 0.0)
        with self.assertRaises(TypeError):
            mb.Scene()

    def test_failed_build_leaves_nothing_behind(self):
        # 全有或全无：非法实例在列表末尾时，前面的合法实例也不生成场景
        good_scene = build_scene([self._inst(1)])
        before = intersect_scene(good_scene, Ray((0.2, 0.3, -1), (0, 0, 1)), 0.0)
        self.assertTrue(before.hit)
        with self.assertRaises(ValueError):
            build_scene([self._inst(1), self._inst(2),
                         self._inst(3, mat=((0, 0, 0), (0, 1, 0), (0, 0, 1)))])
        with self.assertRaises(ValueError):
            build_scene([self._inst(1), self._inst(1)])  # 重复 id
        after = intersect_scene(good_scene, Ray((0.2, 0.3, -1), (0, 0, 1)), 0.0)
        self.assertEqual(after.instance_id, before.instance_id)
        self.assertEqual(after.t, before.t)

    def test_error_is_locatable(self):
        try:
            build_scene([self._inst(1),
                         self._inst(2, mat=((math.nan, 0, 0),
                                            (0, 1, 0), (0, 0, 1)))])
        except ValueError as exc:
            self.assertIn("instances[1]", str(exc))
        else:
            self.fail("应抛出 ValueError")

    def test_query_validation_matches_bvh_contract(self):
        scene = build_scene([self._inst(1)])
        ray = Ray((0.2, 0.3, -1.0), (0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            intersect_scene(scene, Ray((0, 0, 0), (0, 0, 0)), 0.0)
        with self.assertRaises(ValueError):
            intersect_scene(scene, ray, 1.5)
        with self.assertRaises(ValueError):
            intersect_scene(scene, ray, math.nan)
        with self.assertRaises(TypeError):
            intersect_scene(scene, ray, True)
        with self.assertRaises(ValueError):
            intersect_scene(scene, ray, 0.0, t_min=2.0, t_max=1.0)
        with self.assertRaises(ValueError):
            intersect_scene(scene, Ray((math.inf, 0, 0), (0, 0, 1)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
