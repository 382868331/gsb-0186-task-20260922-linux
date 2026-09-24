#!/usr/bin/env python3
"""demo.py — 运动 BVH 的固定输入演示（确定性、离线、纯标准库）。

做两件事：
  1. 正常结果：构建一个确定性运动三角形场景，在多个时刻发射射线，
     用 BVH 与独立暴力参考（逐三角形直接求交）比对，打印二者命中一致，
     以及 BVH 实际少测了多少个面 / 访问了多少节点。
  2. 拒绝边界：真实触发一次输入拒绝（零方向射线），打印抛出的异常。

运行：python demo.py     （约数秒内完成）
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import motion_bvh as mb
from motion_bvh import Triangle, Ray, build, intersect, update
from tests.reference import reference_intersect


def make_scene(n_per_cluster=28, seed=20260922):
    """两簇三角形，分别沿 +z/-z 拉开，并带时间线性运动。"""
    rng = random.Random(seed)
    tris = []

    def cluster(z0, vz, phase):
        out = []
        for i in range(n_per_cluster):
            cx = (i % 7) * 1.2 - 3.6 + 0.1 * math.sin(i + phase)
            cy = (i // 7) * 1.2 - 1.8
            s = 0.45
            a0 = (cx - s, cy - s, z0)
            b0 = (cx + s, cy - s, z0)
            c0 = (cx, cy + s, z0)
            # 每顶点独立线性运动：整体漂移 + 微小呼吸
            drift = (0.6 * math.sin(phase + i * 0.3),
                     0.6 * math.cos(phase + i * 0.21), vz)
            a1 = (a0[0] + drift[0] - 0.05 * i / n_per_cluster,
                  a0[1] + drift[1], a0[2] + drift[2])
            b1 = (b0[0] + drift[0] + 0.05 * i / n_per_cluster,
                  b0[1] + drift[1], b0[2] + drift[2])
            c1 = (c0[0] + drift[0], c0[1] + drift[1] + 0.1,
                  c0[2] + drift[2])
            out.append(Triangle(a0, b0, c0, a1, b1, c1))
        return out

    tris += cluster(12.0, 3.0, 0.0)
    tris += cluster(-12.0, -3.0, 1.7)
    # 再撒一批随机运动三角形填充中间区域
    for _ in range(12):
        def p():
            return (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-6, 6))
        a0, b0, c0 = p(), p(), p()
        a1 = tuple(a0[k] + rng.uniform(-1.5, 1.5) for k in range(3))
        b1 = tuple(b0[k] + rng.uniform(-1.5, 1.5) for k in range(3))
        c1 = tuple(c0[k] + rng.uniform(-1.5, 1.5) for k in range(3))
        tris.append(Triangle(a0, b0, c0, a1, b1, c1))
    return tris


def main():
    t_start = time.perf_counter()
    tris = make_scene()
    n = len(tris)
    bvh = build(tris)
    print(f"场景：{n} 个运动三角形，BVH 构建完成（叶上限 4，8 桶 SAH）")

    # 固定的若干射线与时刻（含瞄准与擦边，代码内确定性生成）
    rng = random.Random(99)
    queries = [
        (Ray((3.0, -0.5, 0.0), (0.05, 0.02, 1.0)), 0.0),
        (Ray((3.0, -0.5, 0.0), (0.05, 0.02, 1.0)), 0.5),
        (Ray((3.0, -0.5, 0.0), (0.05, 0.02, 1.0)), 1.0),
        (Ray((-2.0, 1.0, 2.0), (0.1, -0.08, -1.0)), 0.37),
        (Ray((0.0, 0.0, 0.0), (1.0, 0.3, 0.2)), 0.73),  # 斜穿、含零分量否：非零
        (Ray((20.0, 20.0, 20.0), (-1.0, -1.0, -1.0)), 0.25),
    ]
    for _ in range(6):
        queries.append((
            Ray(tuple(rng.uniform(-6, 6) for _ in range(3)),
                tuple(rng.uniform(-1, 1) for _ in range(3))),
            rng.choice((0.0, 0.25, 0.5, 0.75, 1.0))))

    all_match = True
    total_bvh_tests = total_ref_tests = total_nodes = 0
    hits = 0
    for qi, (ray, tau) in enumerate(queries):
        got = intersect(bvh, ray, tau)
        exp = reference_intersect(tris, ray, tau)
        total_bvh_tests += got.triangles_tested
        total_ref_tests += exp["triangles_tested"]
        total_nodes += got.nodes_visited
        match = got.hit == exp["hit"] and (
            not got.hit or got.face_id == exp["face_id"]
            and abs(got.t - exp["t"]) <= 1e-9 * max(1.0, abs(exp["t"])))
        all_match &= match
        if got.hit:
            hits += 1
        if qi < 5 or not got.hit:
            if got.hit:
                print(f"  查询{qi:02d} τ={tau:<4} 命中 face={got.face_id:<3d} "
                      f"t={got.t:8.4f} (u,v)=({got.u:.4f},{got.v:.4f}) "
                      f"测面 {got.triangles_tested:>3d}/{n} 节点 {got.nodes_visited}"
                      f"  参考一致={match}")
            else:
                print(f"  查询{qi:02d} τ={tau:<4} 未命中（证明无解）"
                      f" 测面 {got.triangles_tested:>3d}/{n} "
                      f"节点 {got.nodes_visited}  参考一致={match}")

    print(f"\n共 {len(queries)} 条射线，命中 {hits} 条；BVH 与独立参考结果全部一致："
          f"{all_match}")
    print(f"三角形测试总数：BVH {total_bvh_tests} vs 暴力 {total_ref_tests}"
          f"（少测 {total_ref_tests - total_bvh_tests} 次，"
          f"仅 {total_bvh_tests / total_ref_tests:.1%}）；"
          f"BVH 节点访问合计 {total_nodes}")

    # 展示一个明确的正常命中结果
    ray0, tau0 = queries[0]
    res0 = intersect(bvh, ray0, tau0)
    p = res0.point
    print(f"\n[正常结果示例] τ={tau0} 射线 o={ray0.origin} d={ray0.direction}")
    print(f"  -> 命中 face_id={res0.face_id}, t={res0.t:.6f}, "
          f"bary=({res0.u:.6f}, {res0.v:.6f}), "
          f"point=({p[0]:.5f},{p[1]:.5f},{p[2]:.5f})")

    # ---- 实际触发的拒绝边界 ---------------------------------------------
    print("\n[拒绝边界] 构造 direction=(0,0,0) 的射线并查询（预期 ValueError）：")
    bad_ray = Ray((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    try:
        intersect(bvh, bad_ray, 0.5)
    except (ValueError, TypeError) as exc:
        print(f"  已捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("零方向射线必须被拒绝")

    dt = time.perf_counter() - t_start
    print(f"\n演示完成，实际耗时 {dt:.2f}s（无 sleep、无预录结果）。")


def run_limit_scale_pass():
    """接口上限规模（300 面）真实计算：批量射线与暴力参考逐条比对。"""
    t0 = time.perf_counter()
    rng = random.Random(20260922 + 300)
    n = mb.MAX_TRIANGLES

    def rpt():
        return (rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-6, 6))

    tris = []
    for _ in range(n):
        a0, b0, c0 = rpt(), rpt(), rpt()
        a1, b1, c1 = rpt(), rpt(), rpt()
        tris.append(Triangle(a0, b0, c0, a1, b1, c1))
    bvh = build(tris)
    rays = []
    for _ in range(240):
        tau = rng.random()
        if rng.random() < 0.6:  # 瞄准随机三角形的随机插值顶点
            t = rng.choice(tris)
            w = rng.choice(("a", "b", "c"))
            p0 = getattr(t, w + "0")
            p1 = getattr(t, w + "1")
            tgt = tuple(p0[k] + tau * (p1[k] - p0[k]) for k in range(3))
            o = tuple(tgt[k] + rng.uniform(-18, 18) for k in range(3))
            d = tuple(tgt[k] - o[k] for k in range(3))
        else:
            o = rpt()
            d = tuple(rng.uniform(-1, 1) for _ in range(3))
        rays.append((Ray(o, d), tau))

    agree = 0
    bvh_tests = ref_tests = nodes = 0
    for ray, tau in rays:
        g = intersect(bvh, ray, tau)
        e = reference_intersect(tris, ray, tau)
        bvh_tests += g.triangles_tested
        ref_tests += e["triangles_tested"]
        nodes += g.nodes_visited
        if g.hit == e["hit"] and (not g.hit or (
            g.face_id == e["face_id"]
            and abs(g.t - e["t"]) <= 1e-9 * max(1.0, abs(e["t"])))):
            agree += 1
    dt = time.perf_counter() - t0
    print(f"\n[上限规模] {n} 面（接口上限）× {len(rays)} 条射线，"
          f"与暴力参考一致 {agree}/{len(rays)}；")
    print(f"  测面总数 BVH {bvh_tests} vs 暴力 {ref_tests}"
          f"（仅 {bvh_tests / ref_tests:.1%}），节点访问合计 {nodes}，"
          f"用时 {dt:.2f}s。")


def run_refit_pass():
    """拓扑保持的顶点更新（refit）演示：正常结果 + 退化边界 + 实际错误。

    场景全部按手算位置构造：两排各 6 个静止等腰三角形分居 z=±10，
    第 i 个上排三角形中心 x=i−2.5（i=0..5）。更新后整体平移 +x 30，
    新盒完全离开旧盒；期望值同时用“全量 build 重建”和独立暴力参考核对。
    """
    print("\n==================== 顶点更新（refit）演示 ====================")

    def row(z):
        out = []
        for i in range(6):
            cx = i - 2.5
            out.append(Triangle.stationary(
                (cx - 0.4, -1.0, z), (cx + 0.4, -1.0, z), (cx, 1.0, z)))
        return out

    tris = row(10.0) + row(-10.0)
    bvh = build(tris)

    # ---- 正常结果：跨旧包围盒平移，查询与全量重建一致 -------------------
    dx = 30.0
    moved = []
    for t in tris:
        shifted = []
        for k in range(6):
            p = getattr(t, ("a0 b0 c0 a1 b1 c1".split())[k])
            shifted.append((p[0] + dx, p[1], p[2]))
        moved.append(Triangle(*shifted))
    update(bvh, moved)
    rebuilt = build(moved)  # 题面基准：同样输入全量重建

    ray_old = Ray((0.5, 0.0, 0.0), (0.0, 0.0, 1.0))   # 瞄准旧位置
    ray_new = Ray((30.5, 0.0, 0.0), (0.0, 0.0, 1.0))  # 瞄准新位置
    miss = intersect(bvh, ray_old, 0.0)
    hit = intersect(bvh, ray_new, 0.0)
    fresh = intersect(rebuilt, ray_new, 0.0)
    ref = reference_intersect(moved, ray_new, 0.0)
    print("[正常] 12 面整体平移 +x30（新盒全部离开旧盒），自底向上 refit 完成")
    print(f"  旧位置射线：命中={miss.hit}，测面 {miss.triangles_tested}"
          f"（根盒已移走，直接剪枝，手算应为 0）")
    print(f"  新位置射线：命中 face={hit.face_id} t={hit.t:.6f}"
          f"（手算：上排第 3 面 id=3、t=10），测面 {hit.triangles_tested}/12")
    print(f"  与全量重建一致：{hit.face_id == fresh.face_id and hit.t == fresh.t}"
          f"；与暴力参考一致：{hit.face_id == ref['face_id']}")

    # ---- 连续第二次更新：上排静止 -> 沿 z 线性运动，下排保持原位；
    #      τ=0.5 上排 z=15 是射线最近命中，手算 t=15 ----------------------
    moving = []
    for i, t in enumerate(moved):
        z1 = 20.0 if i < 6 else -10.0  # 仅上排运动，下排静止在起点后方
        moving.append(Triangle(
            t.a0, t.b0, t.c0,
            (t.a0[0], t.a0[1], z1), (t.b0[0], t.b0[1], z1),
            (t.c0[0], t.c0[1], z1)))
    update(bvh, moving)  # 第二次连续更新
    r = Ray((30.5, 0.0, 0.0), (0.0, 0.0, 1.0))
    h0, h1, hh = (intersect(bvh, r, tau) for tau in (0.0, 1.0, 0.5))
    print(f"[连续两次更新] 第二次改为 z 向运动：τ=0 t={h0.t:.1f}、"
          f"τ=1 t={h1.t:.1f}、τ=0.5 t={hh.t:.1f}（手算 10/20/15）")
    print(f"  τ=0.5 暴力参考 t={reference_intersect(moving, r, 0.5)['t']:.1f}，"
          f"一致={abs(hh.t - 15.0) < 1e-9}")

    # ---- 边界：更新引入退化（零面积）三角形，行为正确 -------------------
    deg = [
        Triangle.stationary(
            (-1.0, -1.0, 4.0), (1.0, -1.0, 4.0), (0.0, 1.0, 4.0)),
        # 共线退化面（更新前是正常面，这里用两棵树直接对照）
        Triangle.stationary(
            (-2.0, 0.0, 7.0), (0.0, 0.0, 7.0), (2.0, 0.0, 7.0)),
    ]
    dbvh = build([Triangle.stationary((9, 9, 9), (9, 9, 9), (9, 9, 9)),
                  Triangle.stationary((8, 8, 8), (8, 8, 8), (8, 8, 8))])
    update(dbvh, deg)
    # (1.5,0) 在正常三角形 x∈[-1,1] 之外，但在共线线段 x∈[-2,2] 上：
    # 退化面不得产生命中
    through_seg = Ray((1.5, 0.0, 0.0), (0.0, 0.0, 1.0))
    through_tri = Ray((0.0, 0.3, 0.0), (0.0, 0.0, 1.0))   # 穿过正常面
    m1 = intersect(dbvh, through_seg, 0.0)
    m2 = intersect(dbvh, through_tri, 0.0)
    r1 = reference_intersect(deg, through_seg, 0.0)
    r2 = reference_intersect(deg, through_tri, 0.0)
    print(f"[边界] 更新后第 2 面为共线退化三角形：穿过它 命中={m1.hit}"
          f"（参考 {r1['hit']}）；正常面仍命中 face={m2.face_id}"
          f" t={m2.t:.1f}（参考 face={r2['face_id']} t={r2['t']:.1f}）")

    # ---- 实际错误：非法更新被拒绝且旧树完整保留 --------------------------
    old_hit = intersect(bvh, r, 0.5)
    bad = list(moving)
    bad[0] = Triangle((math.nan, 0.0, 0.0), bad[0].b0, bad[0].c0,
                      bad[0].a1, bad[0].b1, bad[0].c1)
    print("[错误] 用含 NaN 的新位置更新（预期 ValueError）：")
    try:
        update(bvh, bad)
    except (ValueError, TypeError) as exc:
        print(f"  已捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("含 NaN 的更新必须被拒绝")
    try:
        update(bvh, moving[:-1])  # 数量不符（拓扑不变被破坏）
    except ValueError as exc:
        print(f"  已捕获 {type(exc).__name__}: {exc}")
    kept = intersect(bvh, r, 0.5)
    print(f"  失败后旧树保留：同射线仍命中 face={kept.face_id} "
          f"t={kept.t:.1f}，与失败前一致={(kept.face_id, kept.t) == (old_hit.face_id, old_hit.t)}")


if __name__ == "__main__":
    main()
    run_refit_pass()
    run_limit_scale_pass()
