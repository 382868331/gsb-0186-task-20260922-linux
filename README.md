# motion_bvh — 带运动界的 BVH 射线遍历（纯标准库）

- **Python 版本**：3.14.7（验收版本；仅用标准库 `math` / `dataclasses` / `typing`，
  未在其他版本上验证，不做兼容承诺）
- **依赖**：无第三方依赖，不需要网络、账号、密钥、数据库、Docker 或任何中间件
- **平台**：Linux 原生离线运行
- **内容**：本题专用模块 `motion_bvh.py`、独立暴力参考 `tests/reference.py`、
  必要测试 `tests/test_motion_bvh.py`、固定输入演示 `demo.py`

每个三角形给出 **t=0 与 t=1 两端的三个顶点**，顶点位置在时间 τ∈[0,1] 内
线性运动 `p(τ)=p0+τ·(p1−p0)`。BVH 每个节点保存 **两端 AABB**，查询时对
包围盒端点线性插值（并做保守外扩），再以 Slab 法做射线-盒相交。

## 快速验证

```bash
python -m unittest discover -s tests -v   # 66 个测试，约 0.4s
python demo.py                            # 固定输入演示，约 0.4s，要求 <90s
```

demo 真实计算并打印：
1. 一个正常命中结果（face_id、t、barycentric、point）；
2. 12 条射线 BVH 与独立暴力参考逐条一致，且 BVH 明显少测面；
3. 一个 300 面（接口上限规模）场景 ×240 条射线的批量比对；
4. 一个**实际触发的拒绝边界**：零方向射线抛出 `ValueError`；
5. **实例场景**：同一网格 BVH 的两个实例（旋转+非均匀缩放、镜像+剪切）
   各时刻真实命中，与“三角形显式变换到世界坐标逐面求交”的独立参考
   逐条一致，并打印场景盒/底层盒/三角形三级计数；
6. 一个**实际触发的奇异变换拒绝**：行列式为 0 的实例矩阵抛出 `ValueError`。
无 `sleep`、无预录结果。

## 公开接口

```python
from motion_bvh import Triangle, Ray, build, intersect

tri = Triangle(a0, b0, c0, a1, b1, c1, face_id=None)
tri = Triangle.stationary(a, b, c, face_id=None)   # 静止三角形语法糖
ray = Ray(origin=(x,y,z), direction=(dx,dy,dz))
bvh = build(triangles)                 # -> BVH（构建后只读、可重复查询）
hit = intersect(bvh, ray, time, t_min=0.0, t_max=math.inf)
hit = bvh.intersect(ray, time, t_min=0.0, t_max=math.inf)   # 等价
```

### 数据结构

| 类型 | 字段 |
|---|---|
| `Triangle` | `a0,b0,c0,a1,b1,c1`：各为长度 3 的 `tuple/list` 实数；`face_id` 可选非负 `int`（默认取输入下标 0..N−1，必须互不相同） |
| `Ray` | `origin`、`direction`：长度 3 实数序列；`direction` 允许含零分量，但**不能为零向量**；不必归一化，命中参数 t 以 direction 为单位 |
| `Intersection` | `hit`(bool)、`face_id`、`t`、`u`、`v`（`P=(1−u−v)a+ub+vc`）、`point`、`nodes_visited`、`triangles_tested`；未命中时 t/u/v 为 NaN、point/face_id 为 None，**计数字段仍有效** |

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

## 实例与场景（多实例共享同一棵网格 BVH）

同一网格在场景中重复摆放时，不必复制三角形重新建树：把网格 `build` 一次，
每个摆放位置包一个 `Instance`，再由 `build_scene` 组织场景层包围盒树。
**底层 BVH 被各实例共享**（同一对象引用，不复制面列表、不重新 build）。

```python
from motion_bvh import Instance, build_scene, intersect_scene

bvh = build(triangles)                       # 网格只建一次
ident = ((1,0,0),(0,1,0),(0,0,1))
scene = build_scene([
    Instance(bvh, 1001, rot_scale_matrix, (6.0, 1.0, 2.0)),   # 旋转+非均匀缩放
    Instance(bvh, 1002, mirror_matrix,    (-6.0, -2.0, 5.0)), # 镜像（det<0）
])
hit = intersect_scene(scene, ray, time, t_min=0.0, t_max=math.inf)
hit = scene.intersect(ray, time)             # 等价
```

### 实例数据结构

| 类型 | 字段 |
|---|---|
| `Instance` | `bvh`：`build` 的产物；`instance_id`：场景内唯一非负 `int`（拒绝 bool）；`matrix`：3×3 行主序线性变换，**必须可逆**（允许旋转、非均匀缩放、剪切、镜像/负行列式）；`translation`：长度 3 平移。世界坐标 = `matrix @ 局部 + translation`，变换在时间上固定，网格自身仍按 `time` 运动 |
| `SceneIntersection` | `hit`、`instance_id`、`face_id`（网格内原始面 id）、`t`、`u`、`v`、`point`（**世界坐标**）、`scene_boxes_tested`（场景层盒测试）、`mesh_boxes_tested`（底层 BVH 盒测试合计）、`triangles_tested` |

### 实例语义契约

- **查询**：世界射线按逆变换映到实例局部（`o' = M⁻¹(o−T)`、`d' = M⁻¹d`），
  **方向不归一化**，因此局部与世界的射线参数 t 相同，命中窗口
  `[t_min, t_max]` 与同距带语义在两层一致。返回的命中点变换回世界坐标。
- **最近命中同距决胜**：沿用 16 ulp 同距带；同距时**先比 `instance_id`
  小者，再比 `face_id` 小者**；严格更近者永远优先。两阶段实现，
  与遍历/输入顺序无关。
- **实例包围盒**：取网格两端 AABB（t=0/t=1）各 8 角点经仿射变换后的
  **并集**——保守覆盖整个运动区间，不要求最紧。外扩护栏 =
  `‖M‖_F · g_eps`（网格层重心宽限经线性映射的放大上界，Frobenius 范数 ≥
  最大奇异值）+ 8 ulp 变换舍入。
- **场景层树**：叶为单个实例，按质心最长轴稳定中分；实例盒静态
  （已含整个运动区间），查询时不做时间插值。
- **计数三级区分**：`scene_boxes_tested` / `mesh_boxes_tested` /
  `triangles_tested`，未命中时同样有效（证明无解的遍历量）。
- **空场景**（`build_scene([])` 合法）：任何查询未命中，计数全 0；
  引用空网格 BVH 的实例合法，永不命中、不测三角形。
- **全有或全无**：所有实例先校验再建树——重复 `instance_id`、
  非有限数（矩阵/平移/算出的世界包围盒）、**奇异矩阵**（行列式为 0、
  非有限或下溢为 0）、**逆矩阵含非有限值**（如次正规行列式导致逆元溢出
  为 inf）都抛异常且不留任何部分场景；不放大容差把错误变换混过去。
  错误消息带 `instances[i].matrix[行][列]` 等可定位前缀。

旧接口 `build` / `intersect` / `BVH.intersect` 行为不变，可与场景接口混用。

### 输入上限（接口边界）

- 三角形数 **1 ≤ N ≤ 300**（0 个也合法，表示空场景）；超出抛 `ValueError`。
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
- **全有或全无**：`build` / `build_scene` 先全量校验再构建，非法输入不会
  产生部分对象；`intersect` / `intersect_scene` 为只读查询，不修改树。
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
- **决胜与顺序解耦**：先求严格最小 t 再在同距带内选最小 face_id，
  BVH 与暴力参考（任意顺序）结果一致（开发期 10800 组随机差分 0 不一致）。
- **独立参考**：`tests/reference.py` 独立抄写一份求交与决胜逻辑，
  **不 import、不调用被测核心的任何算法**来取期望值；“往返一致”
  只作为测试中的补充断言。场景参考把每个三角形两端顶点显式变换到
  世界坐标后逐面求交，不借用实例遍历给自己作证。
- **实例层**：场景树只存实例世界包围盒（静态、覆盖整个运动区间），
  网格 BVH 按对象引用共享；查询时射线逆变换到局部后直接复用网格层
  遍历与同距带窗口，跨实例决胜在场景层按 (instance_id, face_id) 完成。

## 已知限制（按本题契约的数学子集，不宣称行业标准兼容）

- SAH 成本只使用 t=0 几何，**不做运动感知划分**；高速运动场景的剪枝效率
  可能次优，但结果正确性对任意 τ 成立。
- 仅支持本文件定义的三角形网格输入与单射线查询；无 Mesh/Material、
  无批量/流式 API、无动态重拟合（refit）、无并行/SIMD 优化。
- 构建后的 BVH 不可变；多线程并发只读查询是安全的，构建本身未做并发设计。
- 完全共面/重合的两个面：几何距离相同即按 face_id 决胜，不做摄动消歧；
  场景层同理，重合实例按 (instance_id, face_id) 决胜。
- `direction` 不归一化：返回的 t 以 direction 长度为单位；实例层把方向
  映到局部也不归一化，故 t 在世界与局部一致。
- 实例变换的数值适用范围：矩阵元、平移与坐标建议在普通 binary64 稳健
  范围内（坐标量级约 1e6 以内，矩阵条件数不过于极端）。行列式下溢为 0
  或逆矩阵元溢出为非有限的变换会被**拒绝**而不是近似放行；极端病态但
  可逆的矩阵（如量级 1e±100 的缩放）可构建，但命中精度随条件数下降，
  未针对此类输入做鲁棒性增强。
- 坐标量级建议在普通 binary64 稳健范围内（约 1e6 以内）；未针对极端
  大坐标/极端细长三角形做鲁棒性增强（如单精度、无除法算法）。

## 文件

```
motion_bvh.py             # 本题核心模块（build/intersect、实例场景层、公开数据结构）
tests/__init__.py         # 测试包标识
tests/reference.py        # 独立暴力参考（逐三角形直接求交；场景参考把三角形
                          # 显式变换到世界坐标后逐面求交，不借用实例遍历）
tests/test_motion_bvh.py  # 网格层必要测试（41 个）
tests/test_instances.py   # 实例/场景层必要测试（25 个）
demo.py                   # 固定输入演示（含实例场景与奇异变换拒绝）
README.md
```
