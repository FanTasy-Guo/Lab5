# 计算机图形学实验五报告

## 光线追踪：折射、软阴影与抗锯齿

---

## 一、实验内容概述

本次实验以经典的 **Whitted-Style 光线追踪模型**为核心，在 Taichi GPU 框架下，通过迭代式光线弹射实现了全局光照效果，涵盖硬阴影、理想镜面反射，并在此基础上完成了折射玻璃材质与抗锯齿的选做扩展。

**基础任务涵盖：**

1. 在 Taichi Kernel 中隐式定义三个几何体（玻璃球、镜面球、棋盘格地面），并建立"材质 ID"系统区分漫反射、镜面反射两类材质；
2. 实现基于迭代 `for` 循环的光线弹射（代替 GPU 不支持的递归），通过 `throughput` 累积能量衰减；
3. 实现 Phong 漫反射着色与**硬阴影**（Shadow Ray），正确处理自相交 Bug（Shadow Acne）；
4. 使用 `ti.ui.Window` 创建交互面板，支持光源位置（X/Y/Z）与最大弹射次数（Max Bounces）的实时滑动调节。

**选做任务在基础任务之上进一步扩展：**

- **选做 1：折射与玻璃材质（+15%）**：引入斯涅尔定律（Snell's Law），将左侧红球改为玻璃材质，根据折射率计算透射方向，并处理全内反射（TIR）与菲涅耳反射率（Schlick 近似）；
- **选做 2：抗锯齿 MSAA（+10%）**：在每个像素内进行 4×4 均匀网格子像素采样，将多条主射线颜色平均，消除物体边缘锯齿。

---

## 二、项目框架

```
├── .venv/                  # Python 虚拟环境
├── src/
│   └── Work0/
│       ├── pycache/
│       ├── lab5.py         # 基础实验（必做部分）
│       ├── ok!.py          # 最终完整版（含全部选做）
│       └── test.py         # 调试测试脚本
├── .gitignore
├── imgui.ini               # ImGui 窗口布局缓存
└── README.md
```

> 核心代码位于 `src/Work0/ok!.py`，实验所有渲染逻辑均在该文件中实现。

---

## 三、基础实验简述

### 3.1 Whitted-Style 光线追踪原理

Whitted 模型将光线路径分为以下几类分支：

$$I = I_{\text{ambient}} + I_{\text{diffuse}} + I_{\text{specular/reflected}}$$

当主射线从摄像机出发击中物体表面时：

| 材质类型 | 处理方式 |
|----------|----------|
| 漫反射（Diffuse） | 按 Phong 模型计算直接光照，终止光线传播 |
| 镜面反射（Mirror） | 计算反射方向，生成次级射线继续弹射 |
| 玻璃折射（Glass） | 按斯涅尔定律计算透射方向，继续追踪 |

### 3.2 场景构建

- **玻璃球**：圆心 $(-1.2,\ 0,\ 0)$，半径 $1.0$，玻璃折射材质（$n=1.5$）；
- **银色镜面球**：圆心 $(1.2,\ 0,\ 0)$，半径 $1.0$，纯镜面反射材质；
- **棋盘格地面**：水平面 $y=-1.0$，通过交点 $x$、$z$ 坐标奇偶性生成黑白格纹理，漫反射材质；
- **摄像机**：固定在 $(0,\ 1,\ 5)$，视角微向下偏移；
- **点光源**：位置由 UI 滑条实时控制，默认 $(2,\ 4,\ 3)$。

### 3.3 迭代式光线弹射

由于 GPU 不支持递归，光线追踪通过 `for` 循环实现：

```python
throughput = [1, 1, 1]   # 光线能量吞吐量
final_color = [0, 0, 0]

for _ in range(max_bounces):
    t, N, color, mat = scene_intersect(ro, rd)
    if 未命中:
        final_color += throughput * bg_color; break
    if 镜面/玻璃:
        更新 ro, rd；throughput *= 衰减系数
    if 漫反射:
        final_color += throughput * 直接光照; break
```

### 3.4 硬阴影与自交修复

从交点向光源发射暗影射线前，必须沿法线方向偏移 $\varepsilon = 10^{-4}$，防止射线与自身表面立刻相交产生 Shadow Acne：

$$\mathbf{P}_{\text{shadow}} = \mathbf{P} + \mathbf{N} \times \varepsilon$$

### 3.5 交互 UI 参数面板

| 参数 | 含义 | 范围 | 默认值 |
|------|------|------|--------|
| Light X / Y / Z | 点光源三维坐标 | $[-5, 5]$ / $[1, 8]$ / $[-5, 5]$ | $(2, 4, 3)$ |
| Max Bounces | 最大光线弹射次数 | $[2, 8]$ | $6$ |
| MSAA Samples | 每像素采样数 | $[1, 16]$ | $1$ |

---

## 四、选做 1：折射与玻璃材质

### 4.1 理论分析

**斯涅尔定律**描述光线在两种介质界面处的折射行为：

$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

向量形式的折射方向公式为：

$$\mathbf{T} = \eta \mathbf{I} + \left(\eta \cos\theta_i - \cos\theta_t\right)\mathbf{N}$$

其中  $\eta = n_1 / n_2$ ， $\cos\theta_i = -\mathbf{I} \cdot \mathbf{N}$ ， $\cos\theta_t = \sqrt{1 - \eta^2(1 - \cos^2\theta_i)}$ 。

当 $\sin^2\theta_t > 1$ 时，发生**全内反射（TIR）**，光线无法折射出去，退化为镜面反射处理。

**Schlick 菲涅耳近似**计算掠射角下的反射率：

$$R(\theta) = R_0 + (1 - R_0)(1 - \cos\theta)^5, \quad R_0 = \left(\frac{n-1}{n+1}\right)^2$$

### 4.2 实现关键点

**法线方向管理**：`intersect_sphere` 始终返回几何外法线（从球心指向交点），由 `trace_ray` 根据 $\mathbf{r}_d \cdot \mathbf{N}$ 的符号判断射线从内部还是外部入射，进而选择正确的 $\eta$ 与法线方向：

```python
from_outside  = rd.dot(N_geo) < 0.0
N_for_refract = N_geo if from_outside else -N_geo
eta           = 1.0 / IOR if from_outside else IOR / 1.0
```

**折射偏移方向**：折射射线穿入另一侧，偏移方向与 `N_for_refract` 相反：

```python
ro = p - N_for_refract * 1e-4   # 沿法线反向偏移，进入介质内部
```

### 4.3 视觉效果

| 特性 | 表现 |
|------|------|
| 玻璃球主体 | 透明折射，可见背后扭曲的棋盘格 |
| 玻璃球边缘 | 菲涅耳效应，掠射角处出现高光亮环 |
| 全内反射区 | 球顶部出现随光源移动的黑色 TIR 区域（物理正确） |
| 镜面球反射 | 反射内容从"红色实心球"变为"扭曲折射的棋盘格" |
| 玻璃球阴影 | 半透明软阴影（透射率约 15%），远浅于不透明物体 |

---

## 五、选做 2：抗锯齿（MSAA）

### 5.1 理论分析

光线追踪中物体边缘锯齿产生的根本原因是每像素只发射一条主射线，无法捕捉亚像素级别的覆盖信息。**多重采样抗锯齿（MSAA）** 在每个像素内均匀发射多条主射线，颜色取平均，使边缘过渡平滑：

$$C_{\text{pixel}} = \frac{1}{N} \sum_{k=1}^{N} C(\mathbf{r}_k)$$

### 5.2 实现方法

采用 4×4 均匀网格子像素采样，采样偏移量计算如下：

```python
col = s % 4;  row = s // 4
dx  = (col + 0.5) / min(ns, 4) - 0.5
dy  = (row + 0.5) / min(ns//4+1, 4) - 0.5
```

当 `MSAA Samples = 1` 时退化为普通单采样，`= 16` 时为完整 4×4 网格覆盖。

### 5.3 视觉效果对比

| MSAA 倍率 | 效果 | 帧率影响 |
|-----------|------|----------|
| 1× | 有明显锯齿（基线） | 无影响 |
| 4× | 边缘明显改善 | 约 ¼ |
| 16× | 边缘极为平滑 | 约 1/16 |

---

## 六、代码逻辑总览

### 6.1 整体渲染管线

```
主循环（Python）
├── render()  ← @ti.kernel（GPU 并行，逐像素执行）
│   │
│   ├── MSAA 子像素采样循环（ns 次）
│   │   └── 生成主射线 (ro, rd)
│   │
│   └── trace_ray()  ← 迭代式光线追踪（max_bounces 次）
│       ├── scene_intersect()  → 最近交点 (t, N, color, mat)
│       ├── MAT_MIRROR  → 更新射线方向，throughput 衰减，继续
│       ├── MAT_GLASS   → 斯涅尔折射 / TIR，throughput 衰减，继续
│       └── MAT_DIFFUSE → soft_shadow_factor() + Phong 着色，break
│
├── canvas.set_image(pixels)
└── GUI 滑动条 → Light XYZ / Max Bounces / MSAA Samples 实时更新
```

### 6.2 关键函数说明

| 函数 | 类型 | 功能说明 |
|------|------|----------|
| `intersect_sphere(ro, rd, center, r)` | `@ti.func` | 解析法求光线-球体交点，返回 $(t,\ \mathbf{N}_{\text{geo}})$ |
| `intersect_plane(ro, rd, plane_y)` | `@ti.func` | 求光线-水平面交点，返回 $(t,\ \mathbf{N})$ |
| `scene_intersect(ro, rd)` | `@ti.func` | 遍历场景取最近交点，返回 $(t, \mathbf{N}, \text{color}, \text{mat})$ |
| `refract_vec(I, N, eta)` | `@ti.func` | 斯涅尔定律计算折射方向，全内反射时返回失败标志 |
| `schlick(cos_theta, ior)` | `@ti.func` | Schlick 近似计算菲涅耳反射率 |
| `shadow_transmittance(ro, rd, dist)` | `@ti.func` | 沿阴影射线累积透射率，玻璃球半透明衰减，不透明物体返回 0 |
| `soft_shadow_factor(p, N, light, r, n)` | `@ti.func` | 区域光圆盘采样，返回平均透射率（软阴影系数） |
| `trace_ray(ro, rd, light, nb)` | `@ti.func` | 迭代式 Whitted 追踪主函数 |
| `render()` | `@ti.kernel` | GPU 并行渲染主核，含 MSAA 采样循环 |

### 6.3 注意事项与调试要点

- **法线统一为几何外法线**：`intersect_sphere` 始终返回朝外法线，内外判断集中在 `trace_ray` 中处理，避免双重翻转导致折射方向混乱；
- **自交偏移方向**：折射穿入时沿法线**反方向**偏移（`p - N * eps`），反射时沿法线**正方向**偏移（`p + N * eps`）；
- **阴影射线未命中判断**：`shadow_t < 0` 表示未击中任何物体，此时不在阴影中，需与 `shadow_t > dist_to_light` 合并判断；
- **Max Bounces 最小值**：玻璃球需要至少 2 次弹射（入射 + 出射），设为 1 时玻璃球内部无法折射出光，显示为全黑；
- **颜色截幅**：最终写入 `pixels` 前使用 `tm.clamp(..., 0.0, 1.0)` 防止过曝。

## 七、效果展示
### 7.1 基础实验效果展示
<div align="center">
  <img src="gif/1.gif" width="700">  
</div>

### 7.2 加入玻璃球和抗锯齿效果
<div align="center">
  <img src="gif/玻璃球.gif" width="700">  
</div>

<div align="center">
  <img src="gif/玻璃球2.gif" width="700">  
</div>
