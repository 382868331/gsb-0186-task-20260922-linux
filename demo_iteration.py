#!/usr/bin/env python3
"""demo_iteration.py — closest_point 增量的真实离线演示（确定性、纯标准库）。

演示内容（全部真实计算，无 sleep、无预录结果、无网络/账号/密钥）：
  1. 正常结果：分散网格场景上的最近点查询，打印 face_id、重心坐标、
     距离平方与表面点；
  2. 与独立逐三角形投影参考（tests/reference.py，不调用被测算法）
     逐条比对一致，并打印盒/面测试计数，展示分支限界确实少测面；
  3. 退化（线段/点）三角形与等距同 face_id 决胜各一例；
  4. 一个实际触发的拒绝边界：非法 time 抛出 ValueError。

运行：python demo_iteration.py     （实测远低于 90 秒）
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_bvh import Triangle, build, closest_point
from tests.reference import reference_closest_point


def make_scattered_grid(n_side=8, spacing=9.0):
    """8x8=64 个彼此远离的运动三角形（分散网格，便于观察剪枝）。"""
    tris = []
    fid = 0
    for i in range(n_side):
        for j in range(n_side):
            cx = i * spacing
            cy = j * spacing
            cz = ((i * 5 + j * 11) % 7) * spacing * 0.4
            s = 0.9
            a0 = (cx - s, cy - s, cz)
            b0 = (cx + s, cy - s, cz)
            c0 = (cx, cy + s, cz)
            a1 = (a0[0] + 0.4, a0[1] - 0.2, a0[2] + 0.3)
            b1 = (b0[0] - 0.1, b0[1] + 0.4, b0[2] - 0.2)
            c1 = (c0[0] + 0.2, c0[1] + 0.1, c0[2] + 0.4)
            tris.append(Triangle(a0, b0, c0, a1, b1, c1, face_id=fid))
            fid += 1
    return tris


def main():
    t_start = time.perf_counter()

    tris = make_scattered_grid()
    bvh = build(tris)
    print(f"场景：{len(tris)} 个分散运动三角形（8x8 网格，间距 9.0）")

    # 1) 一个正常查询结果
    hit = closest_point(bvh, (13.7, 22.4, 5.1), 0.35)
    print("\n[1] 正常查询 point=(13.7, 22.4, 5.1), time=0.35")
    print(f"    face_id={hit.face_id}  dist_sq={hit.dist_sq:.6f}  "
          f"u={hit.u:.4f} v={hit.v:.4f}")
    print(f"    surface point = ({hit.point[0]:.4f}, {hit.point[1]:.4f}, "
          f"{hit.point[2]:.4f})")
    print(f"    nodes_visited={hit.nodes_visited}  "
          f"triangles_tested={hit.triangles_tested} / {len(tris)}")

    # 2) 批量与独立参考比对 + 剪枝计数
    rng = random.Random(20260926)
    n_queries = 120
    mismatches = 0
    sum_tested = 0
    sum_nodes = 0
    for _ in range(n_queries):
        p = (rng.uniform(-5, 75), rng.uniform(-5, 75), rng.uniform(-5, 30))
        tau = rng.random()
        got = closest_point(bvh, p, tau)
        exp = reference_closest_point(tris, p, tau)
        sum_tested += got.triangles_tested
        sum_nodes += got.nodes_visited
        if (got.face_id != exp["face_id"]
                or not math.isclose(got.dist_sq, exp["dist_sq"],
                                    rel_tol=1e-9, abs_tol=1e-9)):
            mismatches += 1
    brute_total = n_queries * len(tris)
    print(f"\n[2] {n_queries} 次随机查询 vs 独立逐三角形投影参考："
          f"不一致 {mismatches} 次")
    print(f"    盒测试(nodes_visited) 平均 {sum_nodes / n_queries:.1f} 次/查询")
    print(f"    面测试(triangles_tested) 合计 {sum_tested} 次，"
          f"暴力参考需 {brute_total} 次 "
          f"（剪枝掉 {100.0 * (1 - sum_tested / brute_total):.1f}% 的面测试）")
    assert mismatches == 0, "BVH 与独立参考结果不一致"
    assert sum_tested < brute_total, "未观察到剪枝"

    # 3) 退化面与等距决胜
    seg = Triangle.stationary((0, 0, 0), (4, 0, 0), (2, 0, 0), face_id=0)
    dot = Triangle.stationary((10, 0, 0), (10, 0, 0), (10, 0, 0), face_id=1)
    bvh_deg = build([seg, dot])
    h_seg = closest_point(bvh_deg, (1.0, 3.0, 4.0), 0.5)
    h_dot = closest_point(bvh_deg, (10.0, 5.0, 0.0), 0.5)
    print("\n[3] 退化面：线段三角形最近点 "
          f"({h_seg.point[0]:.1f}, {h_seg.point[1]:.1f}, {h_seg.point[2]:.1f})"
          f" dist_sq={h_seg.dist_sq:.1f}；"
          f"点三角形 dist_sq={h_dot.dist_sq:.1f}")
    t_hi = Triangle.stationary((0, 0, 5), (2, 0, 5), (0, 2, 5), face_id=9)
    t_lo = Triangle.stationary((0, 0, -5), (0, 2, -5), (2, 0, -5), face_id=3)
    h_tie = closest_point(build([t_hi, t_lo]), (0.5, 0.5, 0.0), 0.5)
    print(f"    等距两面（face_id 9 与 3）决胜：命中 face_id="
          f"{h_tie.face_id}（应为较小的 3）")

    # 4) 实际触发一次拒绝边界
    print("\n[4] 拒绝边界：time=1.5 越界")
    try:
        closest_point(bvh, (0, 0, 0), 1.5)
        print("    未抛出异常（不符合预期）")
    except ValueError as exc:
        print(f"    ValueError: {exc}")

    elapsed = time.perf_counter() - t_start
    print(f"\n演示完成，用时 {elapsed:.2f}s（离线真实计算，无 sleep/预录）")


if __name__ == "__main__":
    main()

