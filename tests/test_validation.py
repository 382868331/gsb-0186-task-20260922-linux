"""输入校验与错误语义测试。"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_bvh import (  # noqa: E402
    MAX_TRIANGLES,
    BudgetExhaustedError,
    MotionBVHError,
    Ray,
    Triangle,
    build,
)

V0 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
V1 = ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))


class ValidationTests(unittest.TestCase):
    def _tri(self, **overrides):
        v0 = overrides.get("v0", V0)
        v1 = overrides.get("v1", V1)
        return (v0, v1)

    def test_nan_and_infinity_rejected_everywhere(self):
        bad_values = (float("nan"), float("inf"), float("-inf"))
        for bad in bad_values:
            with self.subTest(bad=bad, where="vertex"):
                v0 = ((bad, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
                with self.assertRaises(MotionBVHError):
                    build([(v0, V1)])
            with self.subTest(bad=bad, where="ray_origin"):
                with self.assertRaises(MotionBVHError):
                    Ray((bad, 0.0, 0.0), (0.0, 0.0, -1.0))
            with self.subTest(bad=bad, where="ray_direction"):
                with self.assertRaises(MotionBVHError):
                    Ray((0.0, 0.0, 5.0), (1.0, 0.0, bad))
            with self.subTest(bad=bad, where="time"):
                bvh = build([self._tri()])
                with self.assertRaises(MotionBVHError):
                    bvh.intersect(Ray((0.2, 0.2, 5.0), (0.0, 0.0, -1.0)), bad)

    def test_bool_rejected_as_number(self):
        # bool 是 int 的子类，但契约要求拒绝。
        with self.assertRaises(MotionBVHError):
            Triangle(
                ((True, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # type: ignore[arg-type]
                V1,
            )
        ray = Ray((0.2, 0.2, 5.0), (0.0, 0.0, -1.0))
        bvh = build([self._tri()])
        with self.assertRaises(MotionBVHError):
            bvh.intersect(ray, 0.5, budget=True)  # type: ignore[arg-type]
        with self.assertRaises(MotionBVHError):
            Ray((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), tmin=True)  # type: ignore[arg-type]

    def test_bad_shapes_are_localizable(self):
        cases = [
            [("nope", V1)],
            [(V0,)],  # 只给了一端
            [(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), V1)],  # 顶点不是 3 维
            [(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), V1)],  # 只有 2 个顶点
        ]
        for triangles in cases:
            with self.subTest(triangles=triangles):
                with self.assertRaises(MotionBVHError):
                    build(triangles)
        with self.assertRaises(MotionBVHError):
            build("not a sequence")  # type: ignore[arg-type]
        with self.assertRaises(MotionBVHError):
            Ray((0.0, 0.0), (0.0, 0.0, -1.0))  # type: ignore[arg-type]

    def test_error_message_locates_field(self):
        v0 = ((0.0, 0.0, 0.0), (1.0, float("nan"), 0.0), (0.0, 1.0, 0.0))
        with self.assertRaisesRegex(MotionBVHError, r"triangles\[0\]\.vertices0\[1\]\[1\]"):
            build([(v0, V1)])

    def test_triangle_count_limit(self):
        triangles = [
            (((float(i), 0.0, 0.0), (float(i) + 1.0, 0.0, 0.0), (float(i), 1.0, 0.0)),
             ((float(i), 0.0, 1.0), (float(i) + 1.0, 0.0, 1.0), (float(i), 1.0, 1.0)))
            for i in range(MAX_TRIANGLES + 1)
        ]
        with self.assertRaisesRegex(MotionBVHError, str(MAX_TRIANGLES)):
            build(triangles)
        # 恰好在上限处可构建。
        bvh = build(triangles[:MAX_TRIANGLES])
        self.assertEqual(bvh.triangle_count, MAX_TRIANGLES)

    def test_ray_validation(self):
        with self.assertRaisesRegex(MotionBVHError, "non-zero"):
            Ray((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(MotionBVHError, "tmin"):
            Ray((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), tmin=2.0, tmax=1.0)
        with self.assertRaises(MotionBVHError):
            Ray((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), tmax=float("-inf"))

    def test_time_outside_unit_interval(self):
        bvh = build([self._tri()])
        ray = Ray((0.2, 0.2, 5.0), (0.0, 0.0, -1.0))
        for bad_time in (-0.0001, 1.0001, 2.0, -1.0):
            with self.subTest(bad_time=bad_time):
                with self.assertRaises(MotionBVHError):
                    bvh.intersect(ray, bad_time)

    def test_budget_must_be_positive_int(self):
        bvh = build([self._tri()])
        ray = Ray((0.2, 0.2, 5.0), (0.0, 0.0, -1.0))
        for bad in (0, -1, 1.5, "7"):
            with self.subTest(bad=bad):
                with self.assertRaises(MotionBVHError):
                    bvh.intersect(ray, 0.5, budget=bad)  # type: ignore[arg-type]

    def test_build_failure_is_atomic(self):
        good = build([self._tri()])
        bad_scene = [
            self._tri(),
            (
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, float("inf"), 0.0)),
                V1,
            ),
        ]
        with self.assertRaises(MotionBVHError):
            build(bad_scene)
        # 已成功构建的 BVH 仍可正常查询。
        hit = good.intersect(Ray((0.2, 0.2, 5.0), (0.0, 0.0, -1.0)), 0.5)
        self.assertIsNotNone(hit)

    def test_budget_exhausted_is_distinct_from_miss(self):
        # 预算耗尽抛专用异常，绝不能被解读为“证明无命中”。
        scene = [
            (
                ((float(i), 0.0, 0.0), (float(i) + 1.0, 0.0, 0.0), (float(i), 1.0, 0.0)),
                ((float(i), 0.0, 1.0), (float(i) + 1.0, 0.0, 1.0), (float(i), 1.0, 1.0)),
            )
            for i in range(10)
        ]
        bvh = build(scene)
        self.assertGreater(bvh.node_count, 1)
        ray = Ray((0.2, 0.2, 5.0), (0.0, 0.0, -1.0))
        with self.assertRaises(BudgetExhaustedError):
            bvh.intersect(ray, 0.5, budget=1)
        # 同一场景不给预算：完整遍历后命中。
        hit = bvh.intersect(ray, 0.5)
        self.assertIsNotNone(hit)

    def test_empty_scene_is_proven_miss(self):
        bvh = build([])
        self.assertIsNone(bvh.intersect(Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 0.0))

    def test_wrong_types_to_intersect(self):
        bvh = build([self._tri()])
        with self.assertRaises(MotionBVHError):
            bvh.intersect("ray", 0.5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
