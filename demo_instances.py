#!/usr/bin/env python3
"""demo_instances.py — 实例场景查询演示（确定性、离线、纯标准库）。

展示：
  1. 同一棵底层网格 BVH 被两个实例共享（旋转/平移 + 非均匀缩放/镜像），
     不复制面列表、不重新 build；
  2. 真实世界射线查询：返回 instance_id、原 face_id、世界命中点、原局部
     重心坐标；同一网格的两个实例都能被打到；
  3. 计数区分场景盒 / 底层盒 / 三角形测试，并与“把三角形显式变换到世界
     坐标后逐面求交”的独立暴力参考逐条比对；
  4. 一次实际触发的拒绝：奇异（不可逆）矩阵整体拒绝，不生成部分场景；
  5. time 端点（τ=0 与 τ=1）各查一次，说明实例盒覆盖整个运动区间。

运行：python demo_instances.py   （数秒内完成，要求 <90s）
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_bvh import (
    Triangle, Ray, build, Instance, build_scene, intersect_scene,
)
from tests.reference import reference_intersect_scene


def make_shared_mesh():
    """一个小的运动网格（12 个三角形）：t=0 与 t=1 两端整体漂移。"""
    tris = []
    drift0 = (0.0, 0.0, 0.0)
    drift1 = (2.0, 1.0, 0.0)
    for i in range(12):
        cx = (i % 4) * 0.9 - 1.35
        cy = (i // 4) * 0.9 - 0.9
        s = 0.38
        a0 = (cx - s, cy - s, 0.0)
        b0 = (cx + s, cy - s, 0.0)
        c0 = (cx, cy + s, 0.0)
        a1 = (a0[0] + drift1[0], a0[1] + drift1[1], 0.0)
        b1 = (b0[0] + drift1[0], b0[1] + drift1[1], 0.0)
        c1 = (c0[0] + drift1[0], c0[1] + drift1[1], 0.0)
        tris.append(Triangle(a0, b0, c0, a1, b1, c1, face_id=100 + i))
    return tris


def main():
    t_start = time.perf_counter()

    tris = make_shared_mesh()
    # 关键点：网格只 build 一次，两个实例共享同一个 BVH 对象。
    shared_bvh = build(tris)

    # 实例 0：绕 z 轴旋转 ~30°，平移到世界 (x=-4, y=0, z=6)
    ang = math.pi / 6.0
    c, s = math.cos(ang), math.sin(ang)
    M0 = ((c, -s, 0.0),
          (s, c, 0.0),
          (0.0, 0.0, 1.0))
    T0 = (-4.0, 0.0, 6.0)

    # 实例 1：镜像 x（行列式为负）+ 非均匀缩放 + 剪切，平移到 (x=5, y=1, z=6)
    M1 = ((-1.3, 0.4, 0.0),
          (0.0, 0.8, 0.0),
          (0.0, 0.0, 1.6))
    T1 = (5.0, 1.0, 6.0)

    instances = [
        Instance(shared_bvh, 0, M0, T0),
        Instance(shared_bvh, 1, M1, T1),
    ]
    scene = build_scene(instances)
    print(f"共享网格：{len(tris)} 个运动三角形，只 build 了 1 棵底层 BVH；")
    print(f"场景：{scene.instance_count} 个实例引用同一棵树（"
          f"scene._shared_bvh is shared_bvh = {scene._shared_bvh is shared_bvh}）。")

    # 确定性查询：瞄准已知三角形在 τ=0/τ=1 的世界质心（端点），
    # 另加 τ=0.5、明显偏离与斜穿非单位方向的射线。
    def tri_centroid(tr, tau):
        # 三角形三顶点的平均必在三角形内部（u=v=1/3）
        out = []
        for k in range(3):
            s = 0.0
            for w in ("a", "b", "c"):
                v0 = getattr(tr, w + "0")[k]
                v1 = getattr(tr, w + "1")[k]
                s += v0 + tau * (v1 - v0)
            out.append(s / 3.0)
        return tuple(out)

    def world(M, T, p):
        return tuple(M[r][0] * p[0] + M[r][1] * p[1]
                     + M[r][2] * p[2] + T[r] for r in range(3))

    def aim(tri_idx, M, T, tau, d):
        # 取一个保证落在三角形内部的世界目标点，反推起点（非单位方向）
        tgt = world(M, T, tri_centroid(tris[tri_idx], tau))
        tm = 3.0
        origin = tuple(tgt[k] - tm * d[k] for k in range(3))
        return Ray(origin, d)

    d0 = (0.13, -0.21, 1.0)    # 非单位、含横向分量
    d1 = (-0.17, 0.29, 1.0)
    queries = [
        # 实例 0，time 端点 τ=0 与 τ=1（网格漂移后实例盒仍须覆盖）
        (aim(10, M0, T0, 0.0, d0), 0.0),
        (aim(10, M0, T0, 1.0, d0), 1.0),
        # 实例 1（镜像+非均匀缩放），端点 τ=0 与 τ=1
        (aim(5, M1, T1, 0.0, d1), 0.0),
        (aim(5, M1, T1, 1.0, d1), 1.0),
        # τ=0.5 中间时刻
        (aim(2, M0, T0, 0.5, (0.3, 0.1, 1.0)), 0.5),
        # 明显偏离：只测少量场景盒即证明无解
        (Ray((50.0, 50.0, 50.0), (1.0, 0.0, 0.0)), 0.37),
        # 斜穿、非单位方向
        (Ray((0.0, -8.0, 2.0), (0.7, 1.3, 0.9)), 0.5),
    ]

    oracle_input = [(0, tris, M0, T0), (1, tris, M1, T1)]
    all_match = True
    hits = 0
    hit_instances = set()
    sum_scene = sum_under = sum_tri = 0
    print("\n查询（世界射线 → 实例局部，方向不归一化）：")
    for qi, (ray, tau) in enumerate(queries):
        got = intersect_scene(scene, ray, tau)
        exp = reference_intersect_scene(oracle_input, ray, tau)
        match = got.hit == exp["hit"] and (
            not got.hit
            or (got.instance_id == exp["instance_id"]
                and got.face_id == exp["face_id"]
                and abs(got.t - exp["t"]) <= 1e-9 * max(1.0, abs(exp["t"]))))
        all_match &= match
        sum_scene += got.scene_boxes_tested
        sum_under += got.underlying_boxes_tested
        sum_tri += got.triangles_tested
        if got.hit:
            hits += 1
            hit_instances.add(got.instance_id)
            p = got.point
            print(f"  Q{qi} τ={tau:<4} 命中 实例={got.instance_id} "
                  f"原face={got.face_id} t={got.t:7.4f} "
                  f"局部bary=({got.u:.3f},{got.v:.3f}) "
                  f"世界点=({p[0]:6.3f},{p[1]:6.3f},{p[2]:6.3f})")
            print(f"       计数：场景盒 {got.scene_boxes_tested}，"
                  f"底层盒 {got.underlying_boxes_tested}，"
                  f"三角形 {got.triangles_tested}/{2 * len(tris)}（共享，不复制）"
                  f"  参考一致={match}")
        else:
            print(f"  Q{qi} τ={tau:<4} 未命中（证明无解）"
                  f" 场景盒 {got.scene_boxes_tested}，"
                  f"底层盒 {got.underlying_boxes_tested}，"
                  f"三角形 {got.triangles_tested}  参考一致={match}")

    print(f"\n{len(queries)} 条射线，命中 {hits} 条；两个实例均被实际击中："
          f"{hit_instances == {0, 1}}")
    print(f"与世界坐标逐面暴力参考全部一致：{all_match}")
    print(f"计数合计：场景盒 {sum_scene}、底层盒 {sum_under}、"
          f"三角形 {sum_tri}（暴力参考需测 "
          f"{2 * len(tris) * len(queries)} 个世界三角形）")

    # ---- 实际触发的拒绝边界：奇异（不可逆）矩阵 --------------------------
    print("\n[拒绝边界] 尝试把第 3 个实例的线性矩阵设为奇异矩阵（预期 ValueError）：")
    singular = ((1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0),
                (8.0, 10.0, 12.0))  # 第三行 = 2×第二行，行列式 0
    bad_batch = instances + [Instance(shared_bvh, 2, singular, (0.0, 8.0, 6.0))]
    try:
        build_scene(bad_batch)
    except ValueError as exc:
        print(f"  已捕获 ValueError: {exc}")
    else:
        raise AssertionError("奇异矩阵必须被拒绝")
    # 旧场景仍可正常查询（失败批次不产生部分场景、不影响已有对象）
    again = intersect_scene(scene, aim(10, M0, T0, 0.0, d0), 0.0)
    print(f"  拒绝后原场景照常查询：hit={again.hit} "
          f"instance_id={again.instance_id} face_id={again.face_id}")

    dt = time.perf_counter() - t_start
    print(f"\n演示完成，实际耗时 {dt:.2f}s（无 sleep、无预录结果）。")


if __name__ == "__main__":
    main()
