# motion_bvh — 带运动界的 BVH 射线遍历（纯标准库）

- **Python 版本**：3.14.7（验收版本；仅用标准库 `math` / `dataclasses` / `typing`，
  未在其他版本上验证，不做兼容承诺）
- **依赖**：无第三方依赖，不需要网络、账号、密钥、数据库、Docker 或任何中间件
- **平台**：Linux 原生离线运行
- **内容**：本题专用模块 `motion_bvh.py`、独立暴力参考 `tests/reference.py`、
  必要测试 `tests/test_motion_bvh.py` 与 `tests/test_packet_occluded.py`、
  固定输入演示 `demo.py`

每个三角形给出 **t=0 与 t=1 两端的三个顶点**，顶点位置在时间 τ∈[0,1] 内
线性运动 `p(τ)=p0+τ·(p1−p0)`。BVH 每个节点保存 **两端 AABB**，查询时对
包围盒端点线性插值（并做保守外扩），再以 Slab 法做射线-盒相交。

除单射线最近命中 `intersect` 外，模块还提供批量遮挡查询
`packet_occluded`：一批最多 64 条射线**共享节点调度**（每个节点携带仍需
检查的射线位掩码），每条射线有独立的半开 t 区间、time 与 ignore_faces，
只回答“是否被挡住”。

## 快速验证

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 62 个测试，约 0.5s
./.venv/bin/python demo.py                            # 固定输入演示，约 0.3s，要求 <8s
```

demo 真实计算并打印：
1. 一个正常命中结果（face_id、t、barycentric、point）；
2. 12 条射线 BVH 与独立暴力参考逐条一致，且 BVH 明显少测面；
3. 16 条同向成束射线的 `packet_occluded`：与独立逐面遮挡 oracle 逐条一致，
   共享节点调度次数明显少于逐条遍历合计，并演示 ignore_faces 只对所属射线生效；
4. 一个 300 面（接口上限规模）场景 ×240 条射线的批量比对；
5. 两个**实际触发的拒绝边界**：零方向射线单条查询抛 `ValueError`；
   批量遮挡中 `queries[2]` 非法时**整批拒绝**并给出可定位错误。
无 `sleep`、无预录结果。

## 公开接口

```python
from motion_bvh import (
    Triangle, Ray, Intersection, OcclusionQuery, OcclusionPacket,
    build, intersect, packet_occluded,
)

tri = Triangle(a0, b0, c0, a1, b1, c1, face_id=None)
tri = Triangle.stationary(a, b, c, face_id=None)   # 静止三角形语法糖
ray = Ray(origin=(x,y,z), direction=(dx,dy,dz))
bvh = build(triangles)                 # -> BVH（构建后只读、可重复查询）
hit = intersect(bvh, ray, time, t_min=0.0, t_max=math.inf)
hit = bvh.intersect(ray, time, t_min=0.0, t_max=math.inf)   # 等价

# 批量遮挡：最多 64 条，每条独立的半开 t 区间 / time / ignore_faces
q = OcclusionQuery(ray, time=0.5, t_min=0.0, t_max=math.inf,
                   ignore_faces=frozenset())
pkt = packet_occluded(bvh, [q, ray, ...])   # 裸 Ray 取默认参数
pkt = bvh.packet_occluded([q, ...])         # 等价
```

### 数据结构

| 类型 | 字段 |
|---|---|
| `Triangle` | `a0,b0,c0,a1,b1,c1`：各为长度 3 的 `tuple/list` 实数；`face_id` 可选非负 `int`（默认取输入下标 0..N−1，必须互不相同） |
| `Ray` | `origin`、`direction`：长度 3 实数序列；`direction` 允许含零分量，但**不能为零向量**；不必归一化，命中参数 t 以 direction 为单位 |
| `Intersection` | `hit`(bool)、`face_id`、`t`、`u`、`v`（`P=(1−u−v)a+ub+vc`）、`point`、`nodes_visited`、`triangles_tested`；未命中时 t/u/v 为 NaN、point/face_id 为 None，**计数字段仍有效** |
| `OcclusionQuery` | `ray`；`time`∈[0,1]；`t_min`/`t_max` 为该射线自己的**半开**区间 `[t_min, t_max)`；`ignore_faces` 为仅对本射线生效的忽略 face_id 集合（`set/frozenset`，元素非负 int，`None` 视为空） |
| `OcclusionPacket` | `occluded`：按输入位置的 bool 元组；`nodes_visited`（节点调度次数）、`box_tests`（射线-盒测试）、`triangle_tests`（射线-三角形测试）三路独立计数 |

### 语义契约

- `time` 必须在 **[0.0, 1.0]**（端点包含，`τ=0/1` 精确取端点顶点）。
- `t_min ≤ t_max`，区间**双闭**：`t==t_min` 或 `t==t_max` 的命中算入；
  `t_max=+Infinity` 为默认无界（这是唯一被接受的 Infinity，见下）。
- 方向零分量：对应轴退化为“原点必须在板内”判断；双面求交，**不做背面剔除**。
- **最近命中 + 同距按面 ID**：先取严格最小 t*；在与 t* 相差
  `16·eps·max(1,|t|,|t*|)`（eps=2⁻⁵²）的同距带内，取 **face_id 最小**者。
  该规则按“全局最小 t”两阶段实现，**与遍历/输入顺序无关**。
- 零面积（退化）三角形、与射线平行/共面：均视为未命中（det==0）。
- 计数：`nodes_visited` 为执行了盒测试的节点数（含根）；
  `triangles_tested` 为实际做三角形求交的面数。
- 空场景（`build([])` 合法）：任何查询未命中，计数为 0。

### 批量遮挡 `packet_occluded` 语义

- 一批 **1–64 条**查询（0 条也合法，返回空结果）；每条只回答
  **是否被遮挡**（bool），不返回命中点/距离。
- 每条射线的 t 区间为**半开** `[t_min, t_max)`：**`t==t_max` 的接触不算
  遮挡**，`t==t_min` 算入；`t_min==t_max` 为空区间，恒不遮挡（不参与调度）。
- **共享节点调度**：遍历栈元素为 `(节点, 射线位掩码)`，掩码第 i 位表示
  第 i 条射线仍需检查该节点。弹栈时对掩码内每条射线、按**该射线自己的
  time** 插值包围盒做盒测试；未命中某子盒的 lane 只从该分支掩码消失，
  另一分支掩码不受影响（分支掩码求交不会误清未命中 lane）。
- 某条射线一旦找到有效遮挡面（未忽略、半开区间内命中），即整批退出
  其后所有节点（从全局存活掩码清除），其余射线继续。
- `ignore_faces` 是**每条射线各自**的忽略面集合（按 face_id 匹配），
  只影响所属射线；被忽略的面不做三角形测试、不计入计数。
- 计数三路分开：`nodes_visited` 为实际处理的 `(节点,掩码)` 弹栈次数
  （整批共享，同向成束时远小于逐条 `intersect` 之和）；`box_tests` 为
  逐 lane 的射线-盒测试次数；`triangle_tests` 为实际求交的 (射线,面) 对数。
- 结果 `occluded` **按输入位置**返回；任一查询非法则**整批拒绝**（抛异常，
  不返回部分结果），错误消息带 `queries[i]` 位置前缀。
- 默认参数（裸 `Ray` 或缺省字段）下，遮挡布尔与 `intersect(...).hit`
  一致（`[0,+inf)` 内是否有命中）。

### 输入上限（接口边界）

- 三角形数 **1 ≤ N ≤ 300**（0 个也合法，表示空场景）；超出抛 `ValueError`。
- 批量遮挡射线条数 **≤ 64**（位掩码宽度）；超出抛 `ValueError`。
- 叶节点三角形数 **≤ 4**；SAH 固定 **8 桶**。

## 错误语义

- 所有数值标量（顶点、射线原点/方向、time、t_min、t_max）拒绝 **NaN** 与
  **Infinity**：抛 `ValueError`；唯一例外是 `t_max=+Infinity`（无界上界），
  `−Infinity` 仍拒绝。
- 整数参数（`face_id`）必须是 `int`；**`bool` 不作为 int/float 接受**
  （`True/False` 抛 `TypeError`）。
- 类型错误（非 `Triangle`、非 `Ray`、点不是长度 3 的序列、`triangles`
  不可迭代等）抛 `TypeError`；长度错误同样抛 `TypeError`（形状问题）。
- 域错误（time 越界、零方向、t_min>t_max、N>300、负/重复 face_id）抛
  `ValueError`。错误消息带可定位前缀，如 `triangles[12].b0[2] ...`、
  `ray.direction 不能为零向量`。
- **全有或全无**：`build` 先全量校验再构建，非法输入不会产生部分对象；
  `intersect`/`packet_occluded` 为只读查询，不修改 BVH；
  `packet_occluded` 先校验全部最多 64 条查询再遍历，任一非法即整批拒绝。
- **搜索预算**：本实现**没有**遍历预算/迭代上限这一概念——遍历必然终止，
  “未命中”始终是**已证明无解**（盒层次全部测试完），不存在“预算耗尽但
  未知是否有解”的第三种结果；`Intersection` 只有命中/未命中两态。

## 浮点误差口径

不使用“大容差掩盖边界错误”。明确口径如下（全部为 IEEE-754 binary64）：

1. **三角形谓词**：Möller–Trumbore。重心边界仅允许
   `|越界| ≤ 1e-12`（绝对，作用于 u、v、u+v），且只对落在边界内侧做
   **夹取**（clamp 到 0/1），不会把显著越界点报为命中。
2. **包围盒谓词保守性**：为使“盒命中”严格是“盒内某三角形可能命中”的
   必要条件，插值盒在每轴向外扩
   `4·1e-12·D + 4·eps·max(1,|lo|,|hi|)`，其中 D 为该节点两端盒对角线
   的最大分量。第一项覆盖重心边界宽限在物理空间的最坏投影
   （Δu·|e1|+Δv·|e2|，边长 ≤ √3·D），第二项仅抵消 lerp 舍入；
   护栏随盒尺度缩放，**不改变包含关系**（端点盒是端点顶点的精确 min/max）。
3. **同距带**：16 ulp 相对容差（见上），仅用于判定“同距”，不影响严格更近者。
4. **结果精度**：t/u/v/point 由标准浮点运算直接产生，不做迭代修正。
   测试与独立参考的比对阈值为 `|Δt| ≤ 1e-9·max(1,|t|)`、
   `|Δu|,|Δv| ≤ 1e-9`、`|Δpoint| ≤ 1e-8`（双份独立实现的比对口径，
   非边界接受宽限）；**贴边/贴面/顶点/盒角点用例按精确位置构造并要求命中**，
   不靠容差放行。

## 设计取舍

- **构建**：参考时刻 t=0 的质心选轴（最大跨度）→ 固定 8 桶 SAH
  （叶遍历成本 1，成本 `1+(nL·SAL+nR·SAR)/SA父`，取最小平面）；
  仅当 SAH 成本严格小于叶成本且两侧非空时采用，否则进入兜底。
- **退化质心兜底**：当质心包围盒在所有轴退化（如全部三角形同心、共点）、
  SAH 无法给出划分时，按 (主轴质心, 次轴, 第三轴, 输入序号) 稳定排序后
  **中分 n//2**，保证确定性、两侧非空且叶 ≤ 4。
- **运动界**：节点不存“扫掠体凸包”，而是存两端 AABB；查询时插值并外扩，
  对线性顶点运动保守（线性插值的顶点 min/max 被端点 min/max 包含）。
- **遍历**：显式栈（无递归深度风险），先压右后压左；命中后用当前最近
  距离（加同距带）收窄盒窗口；为不漏同距面，剪枝窗口含 tie 带宽。
- **批量遮挡遍历**：单条显式栈承载 `(节点, 64 位射线掩码)`，整批共享
  节点调度；每条 lane 用自己的 time/t 窗口做盒与三角形测试，命中即
  从全局存活掩码清除（`mask &= active` 在弹栈时生效），空半开区间的
  lane 预先以“不遮挡”解决、不占调度。遮挡查询不求最近命中，因此无
  同距决胜逻辑，布尔判定与逐面独立 oracle 完全一致。
- **决胜与顺序解耦**：先求严格最小 t 再在同距带内选最小 face_id，
  BVH 与暴力参考（任意顺序）结果一致（开发期 10800 组随机差分 0 不一致）。
- **独立参考**：`tests/reference.py` 独立抄写一份求交与决胜逻辑，
  **不 import、不调用被测核心的任何算法**来取期望值；“往返一致”
  只作为测试中的补充断言。

## 已知限制（按本题契约的数学子集，不宣称行业标准兼容）

- SAH 成本只使用 t=0 几何，**不做运动感知划分**；高速运动场景的剪枝效率
  可能次优，但结果正确性对任意 τ 成立。
- 仅支持本文件定义的三角形网格输入、单射线最近命中查询与 ≤64 条的批量
  遮挡查询；无 Mesh/Material、无流式 API、无动态重拟合（refit）、
  无并行/SIMD 优化。
- 构建后的 BVH 不可变；多线程并发只读查询是安全的，构建本身未做并发设计。
- 完全共面/重合的两个面：几何距离相同即按 face_id 决胜，不做摄动消歧。
- `direction` 不归一化：返回的 t 以 direction 长度为单位。
- 坐标量级建议在普通 binary64 稳健范围内（约 1e6 以内）；未针对极端
  大坐标/极端细长三角形做鲁棒性增强（如单精度、无除法算法）。

## 文件

```
motion_bvh.py                # 本题核心模块（build/intersect/packet_occluded、公开数据结构）
tests/__init__.py            # 测试包标识
tests/reference.py           # 独立暴力参考（逐三角形直接求交/遮挡 oracle，用于期望值）
tests/test_motion_bvh.py     # 单射线回归测试（41 个）
tests/test_packet_occluded.py# 批量遮挡测试（21 个）
demo.py                      # 固定输入演示
README.md
```
