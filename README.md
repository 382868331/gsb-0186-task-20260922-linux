# motion_bvh — 带运动界的 BVH 射线遍历

为离线渲染加速结构实现的可复用 Python 题库：对一组**逐顶点在时间 t∈[0,1] 上线性
运动**的三角形构建 BVH，在任意查询时刻做最近命中（closest-hit）射线遍历。
节点保存 t=0 与 t=1 两端的 AABB，查询时按时间线性插值。

- **Python 版本**：Python 3.14.7（使用 3.10+ 语法；仅标准库）
- **依赖**：无第三方依赖，不需要安装任何包；核心、测试、演示全部离线运行，
  不访问网络、不需要账号/密钥/数据库/中间件/Docker。
- **几何算法全部自写**，未使用外部几何库或第三方同类实现代替核心算法。

## 运行

```bash
# 必要测试（约 1 秒）
python -m unittest discover -s tests -v

# 固定输入演示（数秒内；本机实测约 0.06 秒）
python demo.py
```

## 接口

```python
from motion_bvh import build, Ray, Triangle

# triangles 中每项：Triangle，或朴素的 (vertices0, vertices1)；
# 每组顶点为 3 个三维点 (a, b, c)，静态三角形两端填相同值。
bvh = build(triangles)                 # -> MotionBVH
hit = bvh.intersect(Ray(o, d, tmin=0.0, tmax=float("inf")), time=0.5)
# 或函数式：intersect(bvh, ray, time)
```

| 名称 | 说明 |
| --- | --- |
| `build(triangles) -> MotionBVH` | 纯函数构建；非法输入抛 `MotionBVHError`，无部分构建结果 |
| `bvh.intersect(ray, time, *, budget=None) -> HitResult | None` | 最近命中；无命中返回 `None`；预算耗尽抛 `BudgetExhaustedError` |
| `bvh.intersect_full(ray, time, *, budget=None) -> (HitResult|None, TraceStats)` | 同上，但 miss 时也能拿到访问计数 |
| `Ray(origin, direction, tmin=0.0, tmax=inf)` | 方向无需归一化；允许零分量，拒绝全零；闭区间 `[tmin,tmax]` |
| `Triangle(vertices0, vertices1)` | 端点顶点；零面积三角形允许构建，但永不命中 |
| `HitResult.face_id / t / barycentric / triangle_tests / nodes_visited` | 面序号、射线参数、(u,v,w) 对应 (a,b,c)、测试面数、访问节点数 |
| `TraceStats.triangle_tests / nodes_visited` | 访问计数（miss 时也可取） |
| 常量 | `MAX_TRIANGLES=300`、`LEAF_MAX=4`、`BIN_COUNT=8` |

**同距规则**：两次命中的射线参数差落在相对带 `2^-45·max(1,|t1|,|t2|)` 内视为
同距，同距簇取较小 `face_id`（与遍历先后顺序无关）；带外严格按距离比较。

## 输入上限与错误语义

- 三角形数量上限 **300**（接口边界，超出拒绝；不做内部分批）。
- 所有数值输入拒绝 **NaN / ±Infinity**（`ray.tmax` 例外：允许有限数或 `+inf`）。
- 整数参数（如 `budget`）拒绝 **bool**（即使 `bool` 是 `int` 子类）。
- 非法输入报 `MotionBVHError`，消息带可定位字段路径，例如
  `triangles[0].vertices0[1][1]: NaN/Infinity is not allowed`，不静默纠正。
- `time` 必须在 `[0,1]`；`budget` 必须为正整数；`tmin<=tmax`。
- **预算耗尽 ≠ 证明无命中**：`BudgetExhaustedError`（携带耗尽前的
  `nodes_visited`/`triangle_tests`）表示搜索未完成；完整遍历后无命中才返回 `None`。

## 浮点误差口径

全部按 IEEE-754 binary64 计算。设 S = max(1, 命中距离)：

- 命中距离 `t` 与独立参考的差异不超过约 `1e-10·S`（测试比对门限；
  两种独立公式实际通常在 1e-13·S 量级）。
- barycentric 与独立参考的差异不超过约 `1e-8`；边/顶点处的理论值
  （0、1/3、1/2 等）实测误差 ≤ 1e-12，输出归一化至和为 1。
- 护栏 `GUARD = 2^-48`（相对量，O(1) 坐标上约十几个 ULP）**只**用于吸收
  “同一数学量的不同运算路径”舍入：查询时刻插值后的 AABB 向外微扩、
  barycentric 边值与 t 区间闭边界允许 GUARD 级越界后夹紧。它不用来掩盖几何
  边界错误：距边 1e-7 的射线仍被正确判为未命中（见
  `test_grazing_ray_just_misses`）。
- 同距带 `2^-45`（约 2.8e-14 相对量）是唯一的“相等距离”容差。

## 设计取舍

- **构建**：自顶向下递归；用 t=0.5 时刻的质心界选最长轴；固定 **8 桶** SAH
  （代价 `nL·AL + nR·AR`，叶代价 `n·A`，均按父表面积归一），前缀/后缀扫描
  一次求出全部 7 个切分代价；SAH 不划算（含父盒零面积）时退化为**稳定中分**：
  按 `(质心坐标, face_id)` 排序后切半，保证质心完全重合时划分仍然确定且
  叶规模恒 ≤ 4。
- **运动界**：节点两端 AABB 各自包住端点几何；中间时刻对两盒线性插值。
  对线性运动的顶点，插值盒恒包含插值几何（逐维
  `(1−t)·min a + t·min b ≤ min_i((1−t)a_i + t b_i)`），是**保守略松**的界，
  换取每节点仅存两盒、查询只做一次插值。
- **遍历**：栈式深度优先；slab AABB 测试（零方向分量按“起点是否在板内”处理，
  支持负方向）；用当前最近距离持续收窄裁剪区间；子节点按分裂轴方向符号决定
  远近，近端先弹。三角形求交为 Möller–Trumbore（双面，端点时刻插值公式保证
  t=0/1 与存储顶点逐位一致）；当前时刻 det=0 的退化/平行面按未命中处理。
- **独立参考**：`tests/reference.py` 不导入本库，用“平面相交 + 边向量度量
  矩阵解重心坐标”的另一条公式逐面扫描；测试期望值全部来自该参考，
  与核心的往返一致仅作补充。
- 构建结果不可再变，遍历本身无可变共享状态。

## 已知限制

- 只实现本题契约的数学子集：线性顶点运动、单射线最近命中、三角形几何；
  不宣称兼容任何行业标准（无加速结构序列化、无任意曲线/实例化/多段运动）。
- 纯 Python 实现，面向 ≤300 面的本题规模；未做多线程或 SIMD 优化。
- 不做 BVH 重拟合/增量更新；运动界是插值保守界而非逐时刻紧界（见上）。
- 本环境安装的 3.14.7 解释器不接受十六进制浮点**字面量**
  （`0x1.0p-48` 语法错误，属该构建的特殊行为），代码统一使用
  `float.fromhex("0x1.0p-48")`，语义相同。

## 目录

```
motion_bvh/__init__.py     # 公开导出
motion_bvh/core.py         # 数据结构、构建、遍历（核心算法）
tests/test_validation.py   # 输入校验与错误语义
tests/test_structure.py    # BVH 结构不变量
tests/test_traversal.py    # 遍历正确性：与独立参考逐条比对
tests/reference.py         # 独立逐面参考（不导入被测核心）
demo.py                    # 固定输入演示
```
