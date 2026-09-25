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
from motion_bvh import (
    Triangle, Ray, build, intersect, Instance, build_scene, intersect_scene,
)
from tests.reference import reference_intersect, reference_scene_intersect


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


def run_instance_pass():
    """实例场景演示：同一网格 BVH 的两个实例（不同仿射变换）真实查询，
    与“三角形显式变换到世界坐标逐面求交”的独立参考比对；
    再真实触发一次奇异变换拒绝。"""
    t0 = time.perf_counter()
    # 网格：一小簇带运动的三角形（只 build 一次）
    tris = make_scene(n_per_cluster=14, seed=7)[:30]
    bvh = build(tris)

    rot = ((0.8, -0.6, 0.0), (0.6, 0.8, 0.0), (0.0, 0.0, 1.0))  # 绕 z 旋转
    scale = ((1.6, 0.0, 0.0), (0.0, 0.7, 0.0), (0.0, 0.0, 1.2))  # 非均匀缩放
    mirror = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))  # x 镜像
    shear = ((1.0, 0.35, 0.0), (0.0, 1.0, 0.2), (0.0, 0.0, 1.0))

    def matmul(A, B):
        return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                           for j in range(3)) for i in range(3))

    mat_a = matmul(rot, scale)
    mat_b = matmul(mirror, shear)
    ta = (6.0, 1.0, 2.0)
    tb = (-6.0, -2.0, 5.0)
    inst_a = Instance(bvh, 1001, mat_a, ta)
    inst_b = Instance(bvh, 1002, mat_b, tb)
    scene = build_scene([inst_a, inst_b])
    print(f"\n[实例场景] 同一网格 BVH（{len(tris)} 面）被 2 个实例共享"
          f"（底层树同一对象：{inst_a.bvh is inst_b.bvh}），"
          f"instance_id=1001（旋转+非均匀缩放）、1002（镜像+剪切）")

    refs = [(1001, mat_a, ta, tris), (1002, mat_b, tb, tris)]

    def world_centroid(tri, mat, trans, tau):
        lc = tuple(sum((tri.a0[k] + tau * (tri.a1[k] - tri.a0[k]),
                        tri.b0[k] + tau * (tri.b1[k] - tri.b0[k]),
                        tri.c0[k] + tau * (tri.c1[k] - tri.c0[k]))[j]
                   for j in range(3)) / 3.0 for k in range(3))
        return tuple(sum(mat[i][k] * lc[k] for k in range(3)) + trans[i]
                     for i in range(3))

    def aimed(tri, mat, trans, tau, back=25.0):
        tgt = world_centroid(tri, mat, trans, tau)
        o = (tgt[0] + 0.3, tgt[1] - 0.2, tgt[2] - back)
        return Ray(o, (tgt[0] - o[0], tgt[1] - o[1], tgt[2] - o[2]))

    queries = [
        (aimed(tris[3], mat_a, ta, 0.0), 0.0),
        (aimed(tris[10], mat_a, ta, 0.5), 0.5),
        (aimed(tris[20], mat_a, ta, 1.0), 1.0),
        (aimed(tris[5], mat_b, tb, 0.25), 0.25),
        (aimed(tris[15], mat_b, tb, 0.75), 0.75),
        (Ray((30.0, 30.0, 30.0), (-1.0, -1.0, -1.0)), 0.6),  # 明显未命中
    ]
    agree = 0
    s_boxes = m_boxes = t_tests = 0
    for qi, (ray, tau) in enumerate(queries):
        got = intersect_scene(scene, ray, tau)
        exp = reference_scene_intersect(refs, ray, tau)
        s_boxes += got.scene_boxes_tested
        m_boxes += got.mesh_boxes_tested
        t_tests += got.triangles_tested
        ok = got.hit == exp["hit"] and (not got.hit or (
            got.instance_id == exp["instance_id"]
            and got.face_id == exp["face_id"]
            and abs(got.t - exp["t"]) <= 1e-9 * max(1.0, abs(exp["t"]))))
        agree += ok
        if got.hit:
            p = got.point
            print(f"  查询{qi} τ={tau:<4} 命中 inst={got.instance_id} "
                  f"face={got.face_id:<3d} t={got.t:8.4f} "
                  f"世界点=({p[0]:6.3f},{p[1]:6.3f},{p[2]:6.3f}) "
                  f"盒(场景/底层)=({got.scene_boxes_tested},{got.mesh_boxes_tested})"
                  f" 测面={got.triangles_tested} 参考一致={ok}")
        else:
            print(f"  查询{qi} τ={tau:<4} 未命中（证明无解）"
                  f" 盒(场景/底层)=({got.scene_boxes_tested},{got.mesh_boxes_tested})"
                  f" 测面={got.triangles_tested} 参考一致={ok}")
    print(f"  与独立参考（三角形显式变换到世界逐面求交）一致 "
          f"{agree}/{len(queries)}；计数合计：场景盒 {s_boxes}、"
          f"底层盒 {m_boxes}、三角形 {t_tests}")

    # ---- 奇异变换拒绝（实际触发） --------------------------------------
    print("\n[拒绝边界] 用奇异矩阵（第 2 行 = 2 × 第 1 行）构建实例：")
    singular = ((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (0.0, 1.0, 0.0))
    try:
        build_scene([Instance(bvh, 2001, singular, (0.0, 0.0, 0.0))])
    except (ValueError, TypeError) as exc:
        print(f"  已捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("奇异矩阵必须被拒绝")

    dt = time.perf_counter() - t0
    print(f"实例演示完成，用时 {dt:.2f}s。")


if __name__ == "__main__":
    main()
    run_limit_scale_pass()
    run_instance_pass()
