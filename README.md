# motion_bvh — 带运动界的 BVH 射线遍历（纯标准库）

- **Python 版本**：3.14.7（验收版本；仅用标准库 `math` / `dataclasses` / `typing`，
  未在其他版本上验证，不做兼容承诺）
- **解释器**：统一使用仓库内 `./.venv/bin/python`（Linux 原生 CPython 3.14.7）
- **依赖**：无第三方依赖，不需要网络、账号、密钥、数据库、Docker 或任何中间件
- **平台**：Linux 原生离线运行
- **内容**：本题专用模块 `motion_bvh.py`、独立暴力参考 `tests/reference.py`、
  必要测试 `tests/test_motion_bvh.py` / `tests/test_packet_occluded.py`、
  固定输入演示 `demo.py`

每个三角形给出 **t=0 与 t=1 两端的三个顶点**，顶点位置在时间 τ∈[0,1] 内
线性运动 `p(τ)=p0+τ·(p1−p0)`。BVH 每个节点保存 **两端 AABB**，查询时对
包围盒端点线性插值（并做保守外扩），再以 Slab 法做射线-盒相交。

模块提供两类查询：**单射线最近命中** `intersect`（输出命中点/barycentric）与
**批量遮挡（阴影射线包）** `packet_occluded`（1..64 条只问是否被挡住，
整批共享一次 BVH 节点调度）。

## 快速验证

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 69 个测试，约 0.4s
./.venv/bin/python demo.py                            # 固定输入演示，约 0.4s，要求 <90s
```

demo 真实计算并打印：
1. 一个正常命中结果（face_id、t、barycentric、point）；
2. 12 条射线 BVH 与独立暴力参考逐条一致，且 BVH 明显少测面；
3. 一个 300 面（接口上限规模）场景 ×240 条射线的批量比对；
4. **packet_occluded**：32 条同向成束阴影射线与逐面独立参考布尔一致，
   共享调度节点数（实测约 23）严格少于逐条遍历之和（实测约 290），
   而盒/三角形测试总数与逐条完全相同；并演示近面忽略→远面遮挡、
   `t_max` 接触不算遮挡；
5. 两个**实际触发的拒绝边界**：单射线零方向抛 `ValueError`；
   packet 中任一射线非法整批拒绝（错误定位到 `rays[i]`）。
无 `sleep`、无预录结果。

## 公开接口

```python
from motion_bvh import (
    Triangle, Ray, ShadowRay, build, intersect, packet_occluded,
)

tri = Triangle(a0, b0, c0, a1, b1, c1, face_id=None)
tri = Triangle.stationary(a, b, c, face_id=None)   # 静止三角形语法糖
ray = Ray(origin=(x,y,z), direction=(dx,dy,dz))
bvh = build(triangles)                 # -> BVH（构建后只读、可重复查询）

# 单射线最近命中（双闭区间 [t_min, t_max]）
hit = intersect(bvh, ray, time, t_min=0.0, t_max=math.inf)
hit = bvh.intersect(ray, time, t_min=0.0, t_max=math.inf)   # 等价

# 批量遮挡（每条各自的半开区间 [t_min, t_max)、time、ignore_faces）
sr = ShadowRay(origin, direction, time=0.0, t_min=0.0,
               t_max=math.inf, ignore_faces=None)
res = packet_occluded(bvh, [sr, ...])          # 1..64 条
res = bvh.packet_occluded([sr, ...])           # 等价
res.occluded          # tuple[bool]：按输入位置一一对应
res.rejection         # tuple：逐条拒绝原因（本实现整体非法即抛错，故恒为 None）
res.nodes_scheduled   # 共享遍历的节点调度次数（整批共享，非逐条之和）
res.box_tests         # 射线×节点盒测试总次数
res.triangle_tests    # 射线×三角形测试总次数
```

### 数据结构

| 类型 | 字段 |
|---|---|
| `Triangle` | `a0,b0,c0,a1,b1,c1`：各为长度 3 的 `tuple/list` 实数；`face_id` 可选非负 `int`（默认取输入下标 0..N−1，必须互不相同） |
| `Ray` | `origin`、`direction`：长度 3 实数序列；`direction` 允许含零分量，但**不能为零向量**；不必归一化，命中参数 t 以 direction 为单位 |
| `Intersection` | `hit`(bool)、`face_id`、`t`、`u`、`v`（`P=(1−u−v)a+ub+vc`）、`point`、`nodes_visited`、`triangles_tested`；未命中时 t/u/v 为 NaN、point/face_id 为 None，**计数字段仍有效** |
| `ShadowRay` | `origin`、`direction`（同 `Ray`）；`time∈[0,1]`（**每条可不同**，默认 0）；`t_min`/`t_max`（**半开**区间，默认 `[0,+∞)`）；`ignore_faces`：该射线专属忽略面 `face_id` 容器（tuple/list/set/frozenset，默认 None） |
| `OcclusionResult` | `occluded`(tuple[bool]，按输入位置)、`rejection`(tuple，逐条拒绝原因；本实现整体非法即抛异常，故成功时恒为 None)、`nodes_scheduled`、`box_tests`、`triangle_tests` |

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

### packet_occluded 语义契约

- **批大小 1..64**（`MAX_PACKET=64`，接口边界；空批或 >64 抛 `ValueError`）。
- 每条 `ShadowRay` 携带**自己的** `time`、半开区间与忽略集合；包围盒按
  **该射线自己的 τ** 插值——同一节点对批内不同射线可得到不同的插值盒。
- **半开区间 `[t_min, t_max)`**（与 `intersect` 的双闭区间刻意区分）：
  `t==t_min` 的接触算遮挡；**`t==t_max` 的接触不算遮挡**；`t_min==t_max`
  必不遮挡。`t_max=+Infinity` 为默认无界（唯一接受的 Infinity）。
- **共享节点调度（不是逐条循环）**：整批一次遍历。栈帧
  `(node, mask)`，`mask` 是该节点仍需检查的射线位掩码；节点弹出时逐 lane
  用各自的 τ/窗口做盒测试得到命中子掩码，叶节点逐 lane×面做半开三角形测试。
- **lane 早退**：某条射线找到第一个有效面后立即从活动掩码清除，
  其后所有节点对它不再发生；其他射线继续。
- **掩码不求交错杀**：某条 lane 未命中某个子树的盒，只会在**该子树内部**
  被跳过；它在兄弟子树帧里的位仍然保留，出栈帧还要与全局活动掩码求交，
  不会因分支掩码运算被误清为“未遮挡”。
- **忽略面只属所属射线**：`ignore_faces` 逐 lane 判断，被忽略的面不对该
  lane 计数、不求交；不影响批内其他 lane。face_id 必须是本 BVH 中存在的
  id，否则该射线非法（整批拒绝）。
- **结果与拒绝按输入位置返回**：`occluded[i]` 即第 i 条；`rejection[i]`
  为逐条拒绝原因（成功为 None）。**任一射线非法 → 整批拒绝**：先全量校验
  再遍历，非法时不返回部分结果（全有或全无），错误定位到 `rays[i]`。
- **三类计数分开**：
  - `nodes_scheduled`：节点**调度**次数——一个节点以非空掩码入栈计一次
    （含根），是整批共享的单一数字；全部 lane 在排队期间已命中的帧弹出时
    不计数；
  - `box_tests`：**射线×节点**盒测试次数（每个仍存活且被派发到该节点的
    lane 计一次，与盒是否命中无关）；
  - `triangle_tests`：**射线×三角形**测试次数（忽略的面不计）。
  对相同射线集合，packet 的盒/三角形测试总数与“逐条单独查询之和”相同
  （同一套谓词与窗口）；`nodes_scheduled` 对成束射线严格更少。
- 空场景：整批不遮挡，三项计数均为 0。

### 输入上限（接口边界）

- 三角形数 **1 ≤ N ≤ 300**（0 个也合法，表示空场景）；超出抛 `ValueError`。
- 叶节点三角形数 **≤ 4**；SAH 固定 **8 桶**。
- packet 批大小 **1 ≤ M ≤ 64**（空批与 >64 抛 `ValueError`）。

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
- packet 额外错误（同样定位到 `rays[i]`）：元素非 `ShadowRay`（含误传
  `Ray`）、批大小不在 1..64、`ignore_faces` 含非 int（含 bool）/负值/
  本 BVH 不存在的 face_id，分别抛 `TypeError`/`ValueError`；**任一条
  非法，整批拒绝**。
- **全有或全无**：`build` 先全量校验再构建，非法输入不会产生部分对象；
  `intersect` 为只读查询，不修改 BVH。`packet_occluded` 同样先整批校验
  （含每条的几何/time/窗口与 ignore_faces 中 id 的存在性），**任一射线
  非法即整批抛错、不开始遍历、不返回部分结果**；成功时不修改 BVH。
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
5. **半开遮挡边界**：packet 的 `t_max` 接触判定用算出的浮点 t 直接比较
   `t >= t_max`（不做对称宽限）；盒测试用同一半开窗口
   （`tfar >= t_min 且 tnear < t_max`），保证“盒不进则面不测”在边界上
   仍成立。构造在 `t_max` 接触的精确用例（整数几何 + Fraction 精确参考）
   要求报**不遮挡**；`t_min` 接触报遮挡。

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
- **决胜与顺序解耦**：先求严格最小 t 再在同距带内选最小 face_id，
  BVH 与暴力参考（任意顺序）结果一致（开发期 10800 组随机差分 0 不一致）。
- **批量遮挡的共享调度**：packet 只做布尔遮挡，不需要跨面比较距离，
  故用“栈帧 + 活动位掩码”的整批单趟遍历取代逐射线栈：节点只被调度一次
  （携带命中父盒的掩码），而不是每条射线各走一遍树。代价是 Python 层
  逐 lane 的位运算与按各自 τ 插值盒；收益是成束射线的节点派发/出栈次数
  显著下降（demo 场景 32 条成束射线 290→23）。射线盒/三角形**谓词调用
  次数与逐条完全相同**——共享的是调度，不是谓词。
- **掩码生命周期**：子帧各持父命中掩码的副本入栈，分支间不共享可变位；
  lane 命中只清全局活动掩码与当前叶的局部掩码；帧弹出以
  `frame_mask & active` 剔除排队期间已在别处命中的 lane，绝不按“本节点
  盒未命中”清理另一分支上的同一位。
- **独立参考**：`tests/reference.py` 独立抄写一份求交与决胜逻辑，
  **不 import、不调用被测核心的任何算法**来取期望值；“往返一致”
  只作为测试中的补充断言。遮挡布尔另有 **Fraction 精确有理数参考**
  （零浮点、零容差，整数几何 + 整数时刻）核对半开边界。

## 已知限制（按本题契约的数学子集，不宣称行业标准兼容）

- SAH 成本只使用 t=0 几何，**不做运动感知划分**；高速运动场景的剪枝效率
  可能次优，但结果正确性对任意 τ 成立。
- 仅支持本文件定义的三角形网格输入；无 Mesh/Material、无动态重拟合
  （refit）、无并行/SIMD 优化。单射线最近命中另有批量**遮挡**接口
  `packet_occluded`（仅布尔结果，不返回命中面/t）；无批量最近命中、
  无流式 packet API。
- packet 共享调度的收益取决于射线相关性：同向成束最明显，完全分散的
  射线仍保证结果正确，但节点调度可能退化到接近逐条量级（谓词工作量不变）。
- 构建后的 BVH 不可变；多线程并发只读查询是安全的，构建本身未做并发设计。
- 完全共面/重合的两个面：最近命中几何距离相同即按 face_id 决胜，
  不做摄动消歧；遮挡查询只报告“是否被挡”，不区分同距的多个面。
- `direction` 不归一化：返回的 t 以 direction 长度为单位。
- 坐标量级建议在普通 binary64 稳健范围内（约 1e6 以内）；未针对极端
  大坐标/极端细长三角形做鲁棒性增强（如单精度、无除法算法）。

## 文件

```
motion_bvh.py                  # 核心模块（build/intersect/packet_occluded、公开数据结构）
tests/__init__.py              # 测试包标识
tests/reference.py             # 独立暴力参考（浮点逐面 + Fraction 精确遮挡 oracle）
tests/test_motion_bvh.py       # 单射线最近命中必要测试（41 个）
tests/test_packet_occluded.py  # 批量遮挡必要测试（28 个）
demo.py                        # 固定输入演示（单射线、上限规模、packet 共享调度）
README.md
```
