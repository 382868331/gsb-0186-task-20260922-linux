"""固定输入演示：带运动界的 BVH 射线遍历。

运行：``python demo.py``（Python 3.14，仅标准库，数秒内完成）。

内容：
1. 用仓库内确定性代码生成一片运动三角形网格（96 面），构建 BVH；
2. 与“逐面直接相交”的独立参考（tests/reference.py，不调用被测核心）逐条射线比对，
   打印一个正常命中结果，并对比 BVH 实际测试面数与暴力法的 96；
3. 触发两类真实边界：输入拒绝（NaN 顶点）与搜索预算耗尽（区别于证明无命中）。
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

from motion_bvh import (  # noqa: E402
    BudgetExhaustedError,
    MotionBVHError,
    Ray,
    build,
)
from reference import reference_intersect  # noqa: E402  (独立参考，不导入被测核心)


def make_scene() -> list:
    """确定性生成 96 个运动三角形：12x8 网格上的小三角，t=0 在 z=0 平面，
    t=1 时整体平移并逐格抖动，运动中仍保持非退化。"""
    rng = random.Random(20260922)
    scene = []
    for j in range(8):
        for i in range(12):
            x, y = float(i), float(j)
            v0 = ((x, y, 0.0), (x + 0.9, y + 0.05, 0.0), (x + 0.05, y + 0.9, 0.0))
            shift_x = rng.uniform(-0.6, 0.6)
            shift_y = rng.uniform(-0.6, 0.6)
            v1 = (
                (x + shift_x, y + shift_y, 3.0),
                (x + 0.9 + shift_x, y + 0.05 + shift_y, 3.0),
                (x + 0.05 + shift_x, y + 0.9 + shift_y, 3.0),
            )
            scene.append((v0, v1))
    return scene


def main() -> int:
    print("=== 带运动界的 BVH 射线遍历演示 ===")
    scene = make_scene()
    n = len(scene)

    t0 = time.perf_counter()
    bvh = build(scene)
    build_seconds = time.perf_counter() - t0
    print(f"场景：{n} 个运动三角形；BVH 节点数 {bvh.node_count}；构建耗时 {build_seconds*1000:.1f} ms")

    # --- 正常结果：一条会命中的射线 -------------------------------------
    query_time = 0.37
    origin = (3.4, 2.3, 6.0)
    direction = (0.07, -0.04, -1.0)
    ray = Ray(origin, direction)

    t0 = time.perf_counter()
    hit = bvh.intersect(ray, query_time)
    bvh_seconds = time.perf_counter() - t0
    expected = reference_intersect(scene, origin, direction, query_time)

    print("\n--- 正常命中 ---")
    print(f"射线 origin={origin} direction={direction} time={query_time}")
    print(f"  BVH : face_id={hit.face_id}  t={hit.t:.10f}  "
          f"barycentric=({hit.barycentric[0]:.6f}, {hit.barycentric[1]:.6f}, {hit.barycentric[2]:.6f})")
    print(f"  参考: face_id={expected['face_id']}  t={expected['t']:.10f}  "
          f"barycentric=({expected['barycentric'][0]:.6f}, {expected['barycentric'][1]:.6f}, "
          f"{expected['barycentric'][2]:.6f})")
    agreement = (
        hit.face_id == expected["face_id"]
        and abs(hit.t - expected["t"]) <= 1e-10 * max(1.0, abs(expected["t"]))
        and all(abs(a - b) <= 1e-8 for a, b in zip(hit.barycentric, expected["barycentric"]))
    )
    print(f"  两者结果一致：{agreement}；BVH 查询耗时 {bvh_seconds*1e6:.1f} us")
    print(f"  实际测试面数：BVH={hit.triangle_tests}（访问节点 {hit.nodes_visited}）"
          f" vs 暴力逐面={n}")

    # 多条射线汇总，证明“少测面”不是个例。
    rng = random.Random(4242)
    bvh_tests = brute_tests = 0
    checked = 0
    for _ in range(24):
        o = (rng.uniform(-1, 12), rng.uniform(-1, 8), rng.uniform(4.0, 7.0))
        d = (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), -1.0)
        h = bvh.intersect(Ray(o, d), query_time)
        e = reference_intersect(scene, o, d, query_time)
        assert (h is None) == (e is None)
        if h is not None:
            assert h.face_id == e["face_id"]
            bvh_tests += h.triangle_tests
            brute_tests += n
            checked += 1
    print(f"  汇总 {checked} 条命中射线：BVH 共测 {bvh_tests} 面 vs 暴力法 {brute_tests} 面，"
          f"平均每条 {bvh_tests / checked:.1f} / {n}")

    # --- 边界 1：输入拒绝（NaN 顶点） -----------------------------------
    print("\n--- 边界：非法输入被拒绝 ---")
    bad_v0 = ((0.0, 0.0, 0.0), (1.0, float("nan"), 0.0), (0.0, 1.0, 0.0))
    bad_v1 = ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))
    try:
        build([(bad_v0, bad_v1)])
    except MotionBVHError as exc:
        print(f"  build() 拒绝含 NaN 的输入：{type(exc).__name__}: {exc}")

    # --- 边界 2：搜索预算耗尽 ≠ 证明无命中 ------------------------------
    print("\n--- 边界：搜索预算耗尽与证明无命中严格分开 ---")
    small_scene = [
        (
            ((float(i), 0.0, 0.0), (float(i) + 0.5, 0.0, 0.0), (float(i), 0.5, 0.0)),
            ((float(i), 0.0, 1.0), (float(i) + 0.5, 0.0, 1.0), (float(i), 0.5, 1.0)),
        )
        for i in range(10)
    ]
    small_bvh = build(small_scene)
    ray2 = Ray((0.25, 0.25, 4.0), (0.0, 0.0, -1.0))
    try:
        small_bvh.intersect(ray2, 0.5, budget=1)
    except BudgetExhaustedError as exc:
        print(f"  budget=1 时：{type(exc).__name__}（已访问 {exc.nodes_visited} 节点，"
              f"已测 {exc.triangle_tests} 面）——不能当作 miss")
    proven = small_bvh.intersect(ray2, 0.5)
    print(f"  不给预算完整遍历：命中 face_id={proven.face_id} t={proven.t:.6f}，"
          f"测试 {proven.triangle_tests} 面")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
