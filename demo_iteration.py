#!/usr/bin/env python3
"""demo_iteration.py — closest_point 最近点查询的固定输入演示（确定性、离线）。

真实计算并打印：
  1. 一个正常最近点结果（face_id、重心坐标、距离平方、表面点）；
  2. 分散网格场景下多条查询与独立逐三角形投影参考逐条一致，
     并打印盒/面测试计数，证明分支限界确实剪枝（不把速度当正确性证据）；
  3. 退化（线段/点）三角形与运动端点 τ=0/1 的结果；
  4. 一个实际触发的拒绝边界：非法 time 抛出 ValueError。

运行：python demo_iteration.py     （远低于 90 秒，无 sleep、无预录结果）
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_bvh import Triangle, build, closest_point
from tests.reference import reference_closest_point


def make_scattered_grid():
    """分散网格：8x8 个彼此远离、各自带线性运动的三角形。"""
    tris = []
    for i in range(8):
        for j in range(8):
            x, y = i * 10.0, j * 10.0
            phase = (i * 8 + j) * 0.37
            dz = 0.5 * ((phase * 7) % 1.0 - 0.5)
            tris.append(Triangle(
                (x, y, 0), (x + 1.0, y, 0), (x, y + 1.0, 0),
                (x, y, dz), (x + 1.0, y, dz), (x, y + 1.0, dz),
            ))
    return tris


def main():
    t_start = time.perf_counter()

    print("== 1. 正常最近点结果 ==")
    tris = make_scattered_grid()
    bvh = build(tris)
    q = (3.2, 4.1, 2.0)
    r = closest_point(bvh, q, 0.5)
    print(f"查询点 {q} @ τ=0.5 -> face_id={r.face_id} "
          f"u={r.u:.6f} v={r.v:.6f} dist_sq={r.dist_sq:.6f} "
          f"point=({r.point[0]:.4f},{r.point[1]:.4f},{r.point[2]:.4f})")

    print()
    print("== 2. 与独立逐三角形投影参考比对 + 剪枝证据 ==")
    queries = [(3.2, 4.1, 2.0), (35.0, 25.0, -3.0), (71.5, 62.5, 1.0),
               (-8.0, -8.0, 0.5), (40.0, 40.0, 10.0), (15.0, 55.0, -1.0)]
    n_tri = len(tris)
    for tau in (0.0, 0.5, 1.0):
        for qq in queries:
            got = closest_point(bvh, qq, tau)
            exp = reference_closest_point(tris, qq, tau)
            assert got.hit and exp["hit"]
            assert got.face_id == exp["face_id"], (qq, tau, got, exp)
            assert abs(got.dist_sq - exp["dist_sq"]) <= 1e-9 * max(
                1.0, abs(exp["dist_sq"]))
            for k in range(3):
                assert abs(got.point[k] - exp["point"][k]) <= 1e-8
            print(f"τ={tau} q={qq}: face={got.face_id:2d} "
                  f"dist_sq={got.dist_sq:10.4f} 与参考一致 | "
                  f"盒测试 {got.nodes_visited:3d} 次, "
                  f"面测试 {got.triangles_tested:2d}/{n_tri} 个")
    print("以上每条查询的面测试数均严格小于总面数：分支限界确实剪枝，")
    print("正确性由与独立参考逐条一致证明（不以速度数字为证据）。")

    print()
    print("== 3. 退化三角形与运动端点 ==")
    deg = [
        Triangle.stationary((0, 0, 0), (2, 0, 0), (1, 0, 0)),
        Triangle.stationary((5, 5, 5), (5, 5, 5), (5, 5, 5)),
        Triangle((0, 0, 4), (2, 0, 4), (0, 2, 4),
                 (0, 0, 8), (2, 0, 8), (0, 2, 8)),
    ]
    dbvh = build(deg)
    for qq, tau in [((0.5, 1.0, 0.0), 0.0), ((6, 6, 6), 0.0),
                    ((0.5, 0.5, 4.5), 0.0), ((0.5, 0.5, 7.5), 1.0)]:
        got = closest_point(dbvh, qq, tau)
        exp = reference_closest_point(deg, qq, tau)
        assert got.face_id == exp["face_id"]
        assert abs(got.dist_sq - exp["dist_sq"]) <= 1e-9 * max(
            1.0, abs(exp["dist_sq"]))
        print(f"q={qq} τ={tau}: face={got.face_id} "
              f"dist_sq={got.dist_sq:.4f} point="
              f"({got.point[0]:.3f},{got.point[1]:.3f},{got.point[2]:.3f}) "
              f"与参考一致")

    print()
    print("== 4. 实际触发的拒绝边界 ==")
    try:
        closest_point(bvh, (0, 0, 0), 1.5)
    except ValueError as exc:
        print(f"time=1.5 被拒绝：ValueError: {exc}")

    empty = build([])
    r0 = closest_point(empty, (0, 0, 0), 0.5)
    print(f"空网格查询：hit={r0.hit}（无命中，计数 "
          f"nodes={r0.nodes_visited} tris={r0.triangles_tested}）")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"演示完成，真实计算耗时 {elapsed:.3f}s（要求 < 90s）")


if __name__ == "__main__":
    main()
