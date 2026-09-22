"""BVH 结构不变量测试：叶规模、端点包围、运动界内插包含、划分完备与确定性。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_bvh import LEAF_MAX, MAX_TRIANGLES, build  # noqa: E402


def _static_tri(i, j=0.0):
    return (
        ((float(i), j, 0.0), (float(i) + 1.0, j, 0.0), (float(i), j + 1.0, 0.0)),
        ((float(i), j, 1.0), (float(i) + 1.0, j, 1.0), (float(i), j + 1.0, 1.0)),
    )


def _collect_leaves(node):
    if node.ids is not None:
        return [node]
    return _collect_leaves(node.left) + _collect_leaves(node.right)


def _all_nodes(node):
    nodes = [node]
    if node.ids is None:
        nodes.extend(_all_nodes(node.left))
        nodes.extend(_all_nodes(node.right))
    return nodes


class StructureTests(unittest.TestCase):
    def test_leaves_at_most_four_and_partition_complete(self):
        scene = [_static_tri(i) for i in range(300)]
        bvh = build(scene)
        self.assertEqual(bvh.triangle_count, 300)
        leaves = _collect_leaves(bvh._root)
        all_ids: list[int] = []
        for leaf in leaves:
            self.assertLessEqual(len(leaf.ids), LEAF_MAX)
            self.assertEqual(list(leaf.ids), sorted(leaf.ids))
            all_ids.extend(leaf.ids)
        self.assertEqual(sorted(all_ids), list(range(300)))

    def test_at_limit_constant(self):
        # 接口上限即 300。
        self.assertEqual(MAX_TRIANGLES, 300)
        bvh = build([_static_tri(i) for i in range(MAX_TRIANGLES)])
        self.assertEqual(bvh.triangle_count, MAX_TRIANGLES)

    def test_endpoint_bounds_contain_triangles(self):
        scene = [_static_tri(i, (i % 7) * 0.5) for i in range(53)]
        bvh = build(scene)
        for node in _all_nodes(bvh._root):
            leaf_ids = [fid for leaf in _collect_leaves(node) for fid in leaf.ids]
            for endpoint, bounds in ((0, node.b0), (1, node.b1)):
                lo, hi = bounds
                for face_id in leaf_ids:
                    verts = scene[face_id][endpoint]
                    for k in range(3):
                        coords = [v[k] for v in verts]
                        self.assertLessEqual(lo[k], min(coords) + 1e-12)
                        self.assertGreaterEqual(hi[k], max(coords) - 1e-12)

    def test_interpolated_motion_bounds_contain_triangles_at_all_times(self):
        # 节点两端 AABB 线性插值后，必须包住插值三角形（运动界一致性）。
        scene = []
        for i in range(40):
            v0 = ((float(i), 0.0, 0.0), (float(i) + 0.5, 0.0, 0.0), (float(i), 1.0, 0.0))
            v1 = ((-float(i), 2.0, 3.0), (-float(i) + 0.5, 2.0, 3.0), (-float(i), 3.0, 3.0))
            scene.append((v0, v1))
        bvh = build(scene)
        for time in (0.0, 0.13, 0.37, 0.5, 0.81, 1.0):
            s = 1.0 - time
            for node in _all_nodes(bvh._root):
                lo = tuple(node.b0[0][k] * s + node.b1[0][k] * time for k in range(3))
                hi = tuple(node.b0[1][k] * s + node.b1[1][k] * time for k in range(3))
                leaf_ids = [fid for leaf in _collect_leaves(node) for fid in leaf.ids]
                for face_id in leaf_ids:
                    for endpoint in range(3):
                        for k in range(3):
                            coord = scene[face_id][0][endpoint][k] * s + scene[face_id][1][endpoint][k] * time
                            self.assertLessEqual(lo[k], coord + 1e-12)
                            self.assertGreaterEqual(hi[k], coord - 1e-12)

    def test_degenerate_centroids_fall_back_to_balanced_mid_split(self):
        # 所有质心完全重合：SAH 无法分桶，稳定中分仍保证叶 <= 4、树平衡。
        v0 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        v1 = ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))
        scene = [(v0, v1) for _ in range(20)]
        bvh = build(scene)
        leaves = _collect_leaves(bvh._root)
        sizes = sorted(len(leaf.ids) for leaf in leaves)
        self.assertLessEqual(max(sizes), LEAF_MAX)
        # 20 个面、每叶 <=4 且中分切半：叶数在 [5, 10] 之间。
        self.assertGreaterEqual(len(leaves), 5)
        self.assertLessEqual(len(leaves), 10)
        # 即便几何完全重合，face_id 仍被完备划分。
        all_ids = [fid for leaf in leaves for fid in leaf.ids]
        self.assertEqual(sorted(all_ids), list(range(20)))

    def test_build_is_deterministic(self):
        scene = [_static_tri((i * 37) % 64, (i * 13) % 17 * 0.25) for i in range(100)]

        def shape(node):
            if node.ids is not None:
                return ("leaf", tuple(node.ids))
            return ("inner", node.axis, shape(node.left), shape(node.right))

        self.assertEqual(shape(build(scene)._root), shape(build(scene)._root))


if __name__ == "__main__":
    unittest.main()
